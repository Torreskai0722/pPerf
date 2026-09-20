# DINO isolated inference diagnosis

Scope: eight executions, four scenes and two MPS modes; 1,848 completed non-warmup inferences. No additional GPU runs or profiling changes. All percentile statistics use every completed observation and NumPy's linear method.

DINO's variation cannot be assigned to one dominant GPU kernel. Time outside GPU kernels is the largest additive covariance contributor in all eight executions (60–88%). The largest host-module-localized part of that time occurs while the decoder's NVTX range is active. This locates a scheduling/dispatch interval; it does not measure CPU execution or prove the decoder's mathematical operations caused it.

| Scene | MPS | P50 ms | P99-P50 ms | Nonkernel r | Decoder-localized gap r | Encoder GPU r |
|---|---|---:|---:|---:|---:|---:|
| scene-0770 | off | 62.805 | 2.374 | 0.771 | 0.594 | 0.526 |
| scene-0398 | off | 62.605 | 2.002 | 0.740 | 0.686 | 0.417 |
| scene-0184 | off | 62.639 | 3.624 | 0.884 | 0.784 | 0.253 |
| scene-0245 | off | 62.967 | 2.771 | 0.889 | 0.840 | 0.398 |
| scene-0770 | on | 63.135 | 2.454 | 0.823 | 0.762 | 0.192 |
| scene-0398 | on | 63.001 | 2.449 | 0.843 | 0.758 | 0.392 |
| scene-0184 | on | 62.776 | 1.960 | 0.773 | 0.726 | 0.432 |
| scene-0245 | on | 62.692 | 3.362 | 0.808 | 0.759 | 0.406 |

## Which kernels?

Among individual exact-name kernel families, encoder `ampere_sgemm_128x64_tn` has the largest positive covariance contribution in every execution. It is called 36 times per inference, totaling approximately 18.3–18.5 ms, but contributes only 2–15% of total latency variance under the additive covariance accounting. Its Pearson r is 0.18–0.49. This is a kernel family aggregated within the encoder, not one launch or one identified neural-network layer.

The decoder's own GPU time is approximately 6.23–6.33 ms and correlates weakly with total latency (r=0.04–0.18). Its host-side gaps are much more strongly associated (r=0.59–0.84). GPU deformable-attention kernels are not dominant variability contributors: encoder aggregate standard deviations are about 0.017–0.023 ms. Every completed inference has 1,280 kernels; all eight runs have the same 272 distinct kernel-name/grid/block/shared-memory signatures. This supports stable launch structure, but does not by itself prove identical memory-access behavior.

## Tail accounting in milliseconds

Components are evaluated on the same interpolated frames defining total P50 and P99. GPU + nonkernel contributions add to the total gap. The decoder-localized column is part of nonkernel time, not an additional component. Values can be negative; these are not independent module percentiles.

| Scene | MPS | Total gap | GPU contribution | Nonkernel contribution | Decoder-localized portion |
|---|---|---:|---:|---:|---:|
| scene-0770 | off | 2.374 | 1.236 | 1.138 | 0.023 |
| scene-0398 | off | 2.002 | 1.404 | 0.598 | 0.363 |
| scene-0184 | off | 3.624 | 0.034 | 3.590 | 3.019 |
| scene-0245 | off | 2.771 | -0.024 | 2.796 | 1.203 |
| scene-0770 | on | 2.454 | 1.454 | 1.000 | 0.056 |
| scene-0398 | on | 2.449 | 0.722 | 1.727 | 0.665 |
| scene-0184 | on | 1.960 | 0.768 | 1.192 | 0.662 |
| scene-0245 | on | 3.362 | 0.161 | 3.201 | 2.024 |

## Concrete startup mechanism

The harness calls `profiler.release_cached_memory()` after its five warmups, and that method calls `torch.cuda.empty_cache()`. Every run shows approximately 3.0–3.5 ms of kernel-free time inside cudaMalloc on its first measured inference, plus approximately 0.10–0.13 ms on the second. This explains a startup allocation contribution despite warmup. The inspected warmup image and the first images of all four scenes are each 1600×900, so this is not evidence of a different warmup image resolution.

This allocation contribution is zero at the interpolated P50/P99 endpoint frames in all eight runs: it explains the first-frame spike, not the reported P99-P50 gaps. `startup_sensitivity.csv` reports correlations with only the first measured frame excluded as a diagnostic; stored samples and official percentiles remain unchanged. Excluding that frame reduces decoder-gap correlations to 0.38–0.79, showing that startup amplifies the full-sample association.

PyTorch documents that empty_cache releases unused allocator cache: https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.empty_cache.html

## Across scenes and matching inputs

The scene-to-scene P50 spread is only 0.362 ms with MPS off and 0.442 ms with MPS on. Tail-gap spreads are 1.623 and 1.402 ms respectively. The largest tail gap occurs in 0184 with MPS off and 0245 with MPS on, rather than one consistently slow input scene.

| Scene | Matching source frames | Cross-mode latency r | Cross-mode r excluding first frame |
|---|---:|---:|---:|
| scene-0770 | 232 | 0.568 | 0.378 |
| scene-0398 | 229 | 0.346 | 0.026 |
| scene-0184 | 232 | 0.313 | 0.031 |
| scene-0245 | 231 | 0.402 | 0.196 |

## What remains unresolved

Observed location: kernel-free intervals, most strongly localized to the decoder's host range. Candidate mechanisms include Python/framework dispatch, allocator behavior, CPU scheduling and driver launch pacing. CUDA copies/memsets occupy only about 0.32–0.33 ms per inference; their covariance contribution is about 0.2–1.2%, providing little support for GPU copy duration as the dominant source. Most variable kernel-free time lies outside recorded CUDA API calls.

The primary traces disable CPU sampling, backtraces and CPU context-switch collection. They cannot distinguish those host mechanisms or resolve finer decoder operations. One execution per scene/mode and temporally dependent frames do not establish an image-content-induced effect. All P99 estimates here are sparse-tail estimates (<1,000 observations). Correlation with a component of total latency is not causal proof. Multiple kernel comparisons are descriptive, with no significance claim.

A causal follow-up would hold the exact image sequence fixed across independent executions and collect CPU stacks/context switches alongside existing CUDA launches. A separately authorized allocator-cache control would test the startup mechanism. No such runs or configuration changes were made.

Reproduce after sourcing ROS and the workspace:

```bash
python3 -m closeloop_analyzer.input_data.dino_diagnosis /mmdetection3d_ros2/analysis_outputs/input2
python3 -m closeloop_analyzer.input_data.dino_diagnosis /mmdetection3d_ros2/analysis_outputs/input2 --host-only
python3 -m closeloop_analyzer.input_data.dino_diagnosis /mmdetection3d_ros2/analysis_outputs/input2 --report-only
```
