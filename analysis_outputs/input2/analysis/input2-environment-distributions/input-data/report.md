# input2-environment-distributions descriptive analysis

## Scope and completion

Scope: **single-run exploratory descriptive**. One execution was used per condition. Frames within an execution are repeated observations, not independent experimental repetitions. No accuracy or confidence claim is made.

Pair campaign: **252/252 valid**; missing or invalid: **0**. Isolated controls were outside this campaign and are not analyzed.

The pair matrix crosses 3 LiDAR models with 6 camera models, 7 same-condition inputs, and MPS modes off, on: 252 cells. Both tenants consume the same clear or adverse bag; labels below compare each adverse cell only with its configured clear baseline.

## Actual input changes

All values below come from decoded payloads in the immutable MCAP bags. Physical-rain conditions are deterministic corruptions of the listed full-duration clear source and retain source timestamps.

### Clear-condition input context

| condition | image luminance mean | image horizontal gradient | LiDAR points | LiDAR mean range (m) |
|---|---:|---:|---:|---:|
| scene-0434 | 101.448 | 1.972 | 34720.000 | 9.376 |
| scene-0245 | 107.267 | 1.733 | 34720.000 | 5.824 |
| scene-0398 | 98.229 | 2.770 | 34720.000 | 8.058 |
| scene-0184 | 105.201 | 2.795 | 34720.000 | 8.507 |
| scene-0738 | 108.277 | 1.923 | 34720.000 | 9.156 |
| scene-0770 | 93.642 | 1.748 | 34720.000 | 8.037 |

### Paired adverse-minus-clear payload changes

`median paired change` is computed at identical source timestamps. Ratios use adverse median divided by matching clear median.

| condition | modality | metric | paired n | clear median | adverse median | median paired change | ratio |
|---|---|---|---:|---:|---:|---:|---:|

## Isolated-model computational response

Not produced: isolated controls were outside the v4 campaign.

| model | adverse condition | median latency change (ms) | clear minimum-P99 range (ms) | adverse minimum-P99 range (ms) | factor | label |
|---|---|---:|---:|---:|---:|---|

The latency table above stores width comparisons; exact median changes and compute-path measurements are in `isolated_mechanism_comparisons.csv`.

### Isolated compute-path summary across adverse conditions

| model | metric | observations | median adverse-clear median change | amplified | damped | unchanged |
|---|---|---:|---:|---:|---:|---:|

### Module execution summary

| model | module | adverse comparisons | median duration change (ms) | amplified widths | damped widths | unchanged widths |
|---|---|---:|---:|---:|---:|---:|

## Controlled synthetic-input validation

This table holds scene, model, weights, MPS-off isolation, and source timestamp alignment fixed while the deterministic weather payload changes. It validates whether computation changed alongside the measured input; it does not establish that a single measured input feature caused the computation change.

| model | condition | input metric | input ratio | latency median change (ms) | latency width factor | GPU-active median change (ms) | kernel-count median change | dynamic metric change |
|---|---|---|---:|---:|---:|---:|---:|---|

## Same-condition pair amplification and damping

Every label in this section compares an adverse pair only with its matching clear pair under the same MPS mode. Isolated widths are not used as denominators.

| target | MPS | adverse pair observations | amplified | damped | unchanged | median width factor | minimum factor | maximum factor |
|---|---|---:|---:|---:|---:|---:|---:|---:|

### Co-runner-specific latency-width response

| pair | target | MPS | conditions | amplified | damped | unchanged | median factor |
|---|---|---|---:|---:|---:|---:|---:|

## Compute, kernel, waiting, co-runner, and MPS mechanisms

`gpu_kernel_active_ms` measures summed kernel service, `kernel_span_ms` measures first-to-last kernel extent, `waiting_ms` is the captured memcpy-adjacent non-kernel interval, and `kernel_count` records launch structure. Their joint movement is descriptive evidence: active-time or count changes are consistent with input-driven work changes; span and waiting changes without matching active-time changes are consistent with scheduling/interference. These measurements do not identify a specific GPU resource.

| target | MPS | metric | observations | median adverse-clear median change | amplified widths | damped widths | unchanged widths |
|---|---|---|---:|---:|---:|---:|---:|

### Intrinsic-versus-pair label agreement

| pair | target | MPS | latency comparisons | same isolated/pair label | different label |
|---|---|---|---:|---:|---:|

### MPS label transitions

MPS rows place the already valid within-mode off/on comparisons side by side; they do not compare raw adverse latency across modes.

| target | off label → on label | observations |
|---|---|---:|

## Architecture-family interpretation boundary

- 3DSSD is point-based: raw point count/range changes can reach preprocessing and early point operations; `sampled_point_count` shows whether its captured final sampling cardinality changed.
- CenterPoint is voxel-based: `occupied_voxel_count` separates raw point loss/addition from the realized sparse voxel workload.
- PointPillars is voxel/pillar-based: `occupied_voxel_count` and its voxel encoder, middle encoder, backbone, neck, and head durations describe the realized pillar workload.
- DeepLabV3+ and ViT-UPerNet are dense segmentation paths with fixed configured resize shapes; input sensitivity is assessed through preprocessing, module durations, kernel activity/count/span, and waiting rather than a proposal count.
- DETR uses a transformer detector path with fixed configured input processing and query structure; the same fixed/dynamic measurements bound any observed response.
- DINO uses the same measured transformer stages: backbone, neck, positional encoding, encoder, decoder, and bbox head. Warm-up must observe every required binding before a run is accepted.
- Faster R-CNN is proposal-based: `proposal_count`, RPN, and ROI module durations expose its captured dynamic post-backbone path.
- YOLOv3 is single-stage: no proposal-stage count is expected, so preprocessing and dense backbone/head GPU measurements are used.

These are architecture-aware interpretations of measured execution structure, not accuracy claims or proof that weather alone caused any co-run latency change.

## Completeness, exclusions, failures, and limitations

No planned run was excluded or invalid.

Excluded execution attempts:

- `/mmdetection3d_ros2/analysis_outputs/input2/runs/failed_attempts/0-scene-fixed-3dssd-detr-attempt-1`: testbed result was not produced; bags started n/a, bags completed n/a, models acknowledged 0. The zero-sample attempt was excluded and the required cell's next attempt succeeded.
- `/mmdetection3d_ros2/analysis_outputs/input2/runs/failed_attempts/0-scene-fixed-3dssd-detr-attempt-2`: no MCAP files for scene 'scene-0434' in /mmdetection3d_ros2/data/input_variation/scene-fixed; bags started 0, bags completed 0, models acknowledged 0. The zero-sample attempt was excluded and the required cell's next attempt succeeded.
- `/mmdetection3d_ros2/analysis_outputs/input2/runs/failed_attempts/1-scene-0252-fixed-3dssd-detr-attempt-1`: interrupted; bags started 0, bags completed 0, models acknowledged 0. The zero-sample attempt was excluded and the required cell's next attempt succeeded.
- `/mmdetection3d_ros2/analysis_outputs/input2/runs/failed_attempts/1-scene-0434-pointpillars-faster-rcnn-attempt-1`: Command '['nvidia-cuda-mps-control']' returned non-zero exit status 1.; bags started n/a, bags completed n/a, models acknowledged 0. The zero-sample attempt was excluded and the required cell's next attempt succeeded.
- `/mmdetection3d_ros2/analysis_outputs/input2/runs/failed_attempts/1-scene-0434-pointpillars-faster-rcnn-attempt-2`: Command '['nvidia-cuda-mps-control']' returned non-zero exit status 1.; bags started n/a, bags completed n/a, models acknowledged 0. The zero-sample attempt was excluded and the required cell's next attempt succeeded.
- `/mmdetection3d_ros2/analysis_outputs/input2/runs/failed_attempts/1-scene-0434-pointpillars-faster-rcnn-attempt-3`: interrupted; bags started 0, bags completed 0, models acknowledged 0. The zero-sample attempt was excluded and the required cell's next attempt succeeded.
- `/mmdetection3d_ros2/analysis_outputs/input2/runs/failed_attempts/1-scene-0434-pointpillars-faster-rcnn-attempt-4`: interrupted; bags started 0, bags completed 0, models acknowledged 0. The zero-sample attempt was excluded and the required cell's next attempt succeeded.
- `/mmdetection3d_ros2/analysis_outputs/input2/runs/failed_attempts/1-scene-0434-pointpillars-faster-rcnn-attempt-5`: interrupted; bags started 0, bags completed 0, models acknowledged 0. The zero-sample attempt was excluded and the required cell's next attempt succeeded.

Limitations:

- One execution per cell supports descriptive comparisons only; within-run frames do not provide run-to-run uncertainty.
- Rain and snow are deterministic synthetic corruptions, not a claim about every natural weather process.
- The study measures computation and latency, not prediction accuracy.
- Pair changes combine both tenants' input responses and co-run effects; without isolated controls they cannot be separated.
- Nsight instrumentation perturbs timing; all matrix cells use the same profiling path, and conclusions remain descriptive.
- No specific shared GPU resource is named without direct evidence.

## Artifacts and regeneration

- Study contract: `/mmdetection3d_ros2/closeloop_perf/studies/input2/study.yaml` (SHA-256 `57ac5175f011ab3f8b915387f9b58d01851ecc168997b581a2c514f9ec8346d1`).
- Pair and new 3DSSD raw runs: `/mmdetection3d_ros2/analysis_outputs/input2/runs/<run_id>`.
- Preserved failed attempts: `/mmdetection3d_ros2/analysis_outputs/input2/runs/failed_attempts/<run-id>-attempt-N`.
- Isolated raw runs are listed in `isolated_validation.csv` when included.
- Analysis tables, plots, manifest, and this report: `closeloop_perf/input2/analysis`.
- Each Goal 1 raw directory retains `config.yaml`, run/model/status evidence, input JSONL, clock evidence, `profile.nsys-rep.gz`, `profile.sqlite.gz`, and `profile_archive.json`. Compression is checksum-verified and lossless.

Run from `/mmdetection3d_ros2` inside `pPerf-host`:

```bash
source /opt/ros/humble/setup.bash
source closeloop_perf/install/setup.bash
python3 -m closeloop_analyzer.input_data.corrected_input_analysis /mmdetection3d_ros2/closeloop_perf/studies/input2/study.yaml --output-root closeloop_perf/input2/analysis --pair-only --violin-reference scene-0252=clear:/mmdetection3d_ros2/analysis_outputs/input1/analysis/input1-two-pass-physical-rain/input-data/frame_metrics.csv
```

Restore one archived trace without replacing the archive:

```bash
gzip -dk RUN/profile.sqlite.gz
gzip -dk RUN/profile.nsys-rep.gz
```

All per-frame, per-module, per-structure, controlled-response, pair context, ECDF, and comparison rows are retained in the CSV artifacts; the tables above summarize rather than silently exclude them.
