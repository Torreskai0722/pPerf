# Controlled environment sensitivity

This study separates each model's own-input sensitivity from changes caused by
its co-runner's input. The authored pairs are 3DSSD/YOLOv3,
CenterPoint/DINO, CenterPoint/YOLOv3, PointPillars/ViT-UPerNet, and
PointPillars/Mask R-CNN. Scenes are ordered 0770, 0398, 0184, 0245; each
condition uses MPS off and on.

The matrix has 56 isolated executions, 40 paired screening executions, and
120 confirmation slots (four cells × five pairs × two modes × three independent
executions). A replay pass is always one. Compatible screening AA/BB executions
occupy confirmation slots, leaving at most 100 new confirmation executions
when all screening runs are compatible. They remain marked as selection
evidence. No additional layer-detail runs are scheduled.

## Commands

Run inside `pPerf-host`, from `/mmdetection3d_ros2`:

```bash
source /opt/ros/humble/setup.bash
source closeloop_perf/install/setup.bash
study=closeloop_perf/studies/input2_crossed/study.yaml
artifacts=/mmdetection3d_ros2/analysis_outputs/input2
bags=/mmdetection3d_ros2/data/input_variation/controlled
manifest="$artifacts/generated_configs/input2-crossed/experiment_manifest.json"
selection="$artifacts/analysis/input2-crossed/input-data/selection.json"

ros2 run closeloop_experiments campaign input-data prepare "$study" \
  --artifact-root "$artifacts" --bag-root "$bags" --phase screening
ros2 run closeloop_experiments campaign input-data run-isolated "$study" \
  --artifact-root "$artifacts"
ros2 run closeloop_experiments campaign input-data run "$study" \
  --artifact-root "$artifacts" --phase screening
ros2 run closeloop_analyzer analyze input-data "$manifest" \
  --output-root "$artifacts" --options '{"phase":"select"}'
ros2 run closeloop_experiments campaign input-data prepare "$study" \
  --artifact-root "$artifacts" --bag-root "$bags" \
  --phase confirmation --selection "$selection"
ros2 run closeloop_experiments campaign input-data run "$study" \
  --artifact-root "$artifacts" --phase confirmation --selection "$selection"
ros2 run closeloop_analyzer analyze input-data "$manifest" \
  --output-root "$artifacts" --options '{"phase":"report"}'
```

`validate-bags` rereads all constructed bags. `--limit N` and `--run-id ID`
allow initial runtime checks to count toward the matrix. Repeating `run` checks
existing evidence and executes missing slots. Failed/invalid artifacts remain
blocked; after correcting the cause, `--retry-blocked` creates a new attempt
with a distinct ID/configuration, preserving the original. `--dry-run` does
not reserve runtime output directories. Do not run two campaign writers
concurrently against one manifest or GPU.

## Inputs and invariants

The frozen inclusive window is `[first bag timestamp, first + 19.250761 s]`.
Preparation verifies that this is the shortest complete source duration.
LiDAR comes from the first ordered scene; camera comes from the second.
The source-specific shift takes the source's first bag timestamp to positive
origin 1,000 seconds. The same shift is applied to bag and ROS header times,
preserving bag/header offsets, all intervals, payloads, metadata, and frame IDs.

The installed `rosbag2_py` MCAP reader/writer performs I/O. Header stamps are
patched in the checked CDR header layout, preserving padding and every other
serialized byte; Fast-CDR reserialization does not guarantee stable padding.
Ties in bag time are ordered LiDAR then camera, retaining per-stream source
order. A content/settings identity deduplicates bags independently of models,
MPS, and repetition. Only four same-scene bags are initially constructed;
confirmation constructs missing ordered A/B combinations.

Each bag directory contains `manifest.json`, `source_frames.jsonl`, and
`bag/`. Manifests record source/output hashes, counts, transformations, scene
identities, topic metadata, and full read-back validation. The source index
records original/output bag/header times, source bag/scene/ordinal/frame ID,
payload and serialized checksums. Per-stream validation hashes enforce
unchanged streams across all four cells. Source files are never modified.

Camera models use CPUs 0–5, LiDAR models 6–11, including isolation; each
uses three threads. Replay uses CPUs 12–15 and four threads. Existing model
weights, preprocessing, five warmups, launch offsets, readiness and completion
coordination, rate 1.0, best-effort depth-one queues, primary Nsight setup,
clock requests 3105/10501 MHz, and MPS 100% are retained.

## Evidence and analysis

The experiment manifest lives under `generated_configs/input2-crossed/`.
It retains complete slot identities, repetition, planned/observed execution
order, configuration/model hashes, bag/artifact paths, historical reuse
decisions and exact rejection reasons, attempts, and status. Historical files
are read-only inputs. Successful status alone is not evidence of compatibility.
Full-scene runs are never trimmed retrospectively to pass the common-window
protocol. Archive checks verify compressed hashes, gzip integrity, restored
hashes, and SQLite integrity. SQLite restoration for analysis uses temporary
files outside retained run directories.

`selection.json` requires both models' valid four-scene screening results.
The original campaign froze untrimmed linear-percentile P50, P99, P99−P50, explicit
`R=(P99−P50)/P50`, counts, throughput, warnings, model ranges, and source-run
hashes. It selects the larger model R range, then that model's minimum/maximum
scenes. Exact ties follow authored model/scene order, with distinct A/B when
all scenes tie. Selection is immutable. The existing `normalized_range`
statistic continues to mean `(P99−P0)/P50` in older analyses.

The first cell letter names the LiDAR input; the second names the camera
input. Missing runs follow AA/AB/BB/BA, BA/BB/AB/AA, then AB/AA/BA/BB across
repetition blocks, with rotated pair/mode blocks. Reused historical executions
keep their observed historical ordering.

Per-run results use `analysis/runs/<actual-run-id>/input-data/`: completed
inference frames, source-frame coverage, unmatched records, summaries, and
compressed kernel CSV exports. Combined results use
`analysis/input2-crossed/input-data/`: selection, individual-run summaries,
four-cell tables/plots, contrasts and execution-level mean/sample SD/range,
isolated effect comparisons, common-processed-frame comparisons, MPS
comparisons for matching actual scenes, archive indexes, warnings, and report.
`isolated_baselines.md` displays the single-model baseline violin plots: scene
on the x-axis and CUDA-complete inference time (ms) on the y-axis, with adjacent
MPS-off/on violins on a single plot for each model. Tick labels show only scene
names. All current plots and analysis metrics use the inclusive original
P1–P99 latency interval per execution/model/actual scene combination. Cutoffs
use NumPy's linear method; P50, P99, P99−P50, R, means, variability, and
correlations are recomputed on retained frames. Components use those same
frame identities. Filter once before pooling; matched comparisons intersect
the retained frame sets without another percentile crop. Execution-level
replicates are not trimmed. Raw recordings and full exports remain intact.
Sample/unique-frame counts and cutoffs are in `isolated_violin_display.csv`. Plots are
exported under `plots/isolated/` as seven PNGs and one multipage PDF. The normal
`--options '{"phase":"report"}'` command regenerates them from validated evidence.

Bag messages are expected inputs. Only logged relay publications are measured
publications. Missing bag-to-relay records are upstream missing coverage;
published inputs absent from callbacks are dropped/overwritten, without
claiming an unobserved queue mechanism. Completed non-warmup inference NVTX
ranges must contain CUDA synchronization. Decode/preprocessing stay separate.
`throughput_hz` now uses retained count divided by the original
replay-resume-to-later-of-window-end-or-last-completion duration;
`observed_throughput_hz` and `completed_count` preserve actual execution
coverage. Elapsed and drain times remain unchanged. Summaries record
`analysis_count`, `analysis_unique_source_count`, original cutoffs, and excluded
source/input IDs. Empty retained samples block statistical analysis.

Historical `selection.json` and the experiment manifest remain immutable:
they identify the A/B scenes actually executed. `selection_p1_p99.json`
records screening choices recomputed under the new policy, without changing
completed condition identities or scheduling new executions. Superseded
reports and per-run exports are retained in
`generated_configs/input2-crossed/validation/before-p1-p99/`. The shared
`closeloop_analyzer.sample_filter` implements this policy for future analysis.

Kernel exports retain process/context/stream identity, launch correlations,
and only supported model/input/module labels, with attribution coverage by
kernel count and summed GPU duration including unresolved records. Depth-zero
annotations cannot resolve full layer attribution; layer/module ranges and
correlated launches would be needed, but no extra detail runs are scheduled.

P99 below 100 observations is especially fragile; below 1,000 is a sparse-tail
estimate. Sequential frames are temporally dependent. Repetitions are never
pooled as independent frames. Targeted longer measurements may be recommended
without expanding this matrix. Assess 3DSSD and YOLOv3 independently; no
insensitive-control claim follows merely from a small observed difference.

## Validation

```bash
python3 -m pytest \
  closeloop_perf/src/closeloop_experiments/test/test_input_crossed.py \
  closeloop_perf/src/closeloop_analyzer/test/test_crossed_analysis.py \
  closeloop_perf/src/closeloop_experiments/test/test_package_boundaries.py
cd closeloop_perf
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

The final cross-package validation reported 528 tests, two failures, and one
skip. Both failures reference the absent unrelated
`studies/section4_1/study.yaml` from the existing signed-offset-sweep tests.
The 59 focused checks passed, including real MCAP read/write/read-back,
matrix/repetition/resource, frozen selection, archive rejection, resume,
twenty-execution AA/BB reuse, and model-specific replay identity tests.
Initial runtime validation also found and fixed a
missing `message_header_timestamp_ns` import in the shared model callback;
the failed attempt remains retained. Two initial validation executions that
overlapped focused tests were excluded in `execution_exclusions.json` and
replaced without concurrent testing. All original attempts remain retained.
Offline screening validation also rejected one PointPillars/ViT-UPerNet
attempt whose final CUDA synchronization records were missing after a forced
process shutdown. Its replacement passed; all ten selections then froze
without blockers. The nine final analyzer checks passed, including authored
scene tie ordering and retention of excluded attempts in the archive index.

The completed input2 execution has 216 validated slots backed by 196 distinct
compatible executions, with 20 screening reuses and exactly 100 new confirmation
executions. All 16 bags and ten selections are retained. Offline validation
reported zero condition blockers and produced 20 four-cell plots and 196 kernel
exports. Scientific findings are recorded in
`analysis/input2-crossed/input-data/findings.md` under the supplied artifact root,
alongside the complete generated report and tables. The findings assess 3DSSD
and YOLOv3 separately and do not label their pair an insensitive control.
