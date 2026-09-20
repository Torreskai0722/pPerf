# DINO isolated inference diagnosis — P1–P99 filtered

Eight executions, four scenes, two MPS modes; 1800 retained inference observations. Cutoffs are computed once per execution from original completed non-warmup latencies. All statistics, including P50/P99, are recomputed after filtering. Module, kernel and host measurements use the same retained frames. Original cutoffs and excluded source/input IDs are in summary.json. No additional GPU runs were performed.

| Scene | MPS | Retained | P50 ms | P99−P50 ms | Nonkernel r | Decoder host-gap r | Encoder GPU r |
|---|---|---:|---:|---:|---:|---:|---:|
| scene-0770 | False | 226 | 62.805 | 1.885 | 0.686 | 0.408 | 0.599 |
| scene-0398 | False | 223 | 62.605 | 1.159 | 0.602 | 0.452 | 0.343 |
| scene-0184 | False | 226 | 62.639 | 1.620 | 0.746 | 0.653 | 0.356 |
| scene-0245 | False | 225 | 62.967 | 2.004 | 0.817 | 0.762 | 0.430 |
| scene-0770 | True | 226 | 63.135 | 1.643 | 0.666 | 0.479 | 0.401 |
| scene-0398 | True | 223 | 63.001 | 2.068 | 0.762 | 0.675 | 0.418 |
| scene-0184 | True | 226 | 62.776 | 1.425 | 0.618 | 0.515 | 0.528 |
| scene-0245 | True | 225 | 62.692 | 1.053 | 0.660 | 0.628 | 0.313 |

## Kernel and host associations

Covariance shares are descriptive additive contributions to observed latency variance; correlation with a component of total latency does not establish causation. A kernel family aggregates exact-name launches inside one recorded module, not one layer.

| Scene | MPS | Nonkernel covariance share | Largest kernel-family covariance share | Module | Kernel |
|---|---|---:|---:|---|---|
| scene-0770 | False | 0.466 | 0.102 | encoder | `ampere_sgemm_128x64_tn` |
| scene-0398 | False | 0.523 | 0.057 | backbone | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` |
| scene-0184 | False | 0.723 | 0.056 | encoder | `ampere_sgemm_128x64_tn` |
| scene-0245 | False | 0.715 | 0.088 | encoder | `ampere_sgemm_128x64_tn` |
| scene-0770 | True | 0.611 | 0.109 | encoder | `ampere_sgemm_128x64_tn` |
| scene-0398 | True | 0.673 | 0.120 | encoder | `ampere_sgemm_128x64_tn` |
| scene-0184 | True | 0.550 | 0.118 | encoder | `ampere_sgemm_128x64_tn` |
| scene-0245 | True | 0.566 | 0.050 | encoder | `ampere_sgemm_128x64_tn` |

Kernel-free intervals include host work, dispatch gaps, CUDA API waits, and device copies. Host NVTX ranges locate these intervals; they do not identify CPU execution time. The external decode/preprocess measurements remain outside inference. Internal data_preprocessor work is inside inference and is reported separately from decoder/encoder gaps.

| Scene | MPS | Outside CUDA API ms | Launch API ms | Stream synchronize ms | Copy/memset in gaps ms | Internal preprocessing gap ms | Decoder gap ms |
|---|---|---:|---:|---:|---:|---:|---:|
| scene-0770 | False | 5.953 | 1.366 | 0.553 | 0.332 | 0.898 | 3.552 |
| scene-0398 | False | 5.861 | 1.349 | 0.547 | 0.332 | 0.865 | 3.497 |
| scene-0184 | False | 6.009 | 1.400 | 0.541 | 0.326 | 0.886 | 3.599 |
| scene-0245 | False | 6.186 | 1.430 | 0.543 | 0.329 | 0.915 | 3.727 |
| scene-0770 | True | 6.079 | 1.399 | 0.548 | 0.320 | 0.875 | 3.645 |
| scene-0398 | True | 6.081 | 1.417 | 0.549 | 0.333 | 0.886 | 3.671 |
| scene-0184 | True | 5.996 | 1.440 | 0.546 | 0.325 | 0.906 | 3.626 |
| scene-0245 | True | 5.933 | 1.384 | 0.549 | 0.328 | 0.936 | 3.546 |

These categories overlap: copy duration and host-module locations must not be added to the CUDA API partition.

## Tail accounting

All components use the same interpolated retained frames defining total P50/P99. GPU + nonkernel contributions add to the total gap; the decoder portion is included in nonkernel time. Contributions can be negative and are not independently computed component percentile gaps.

| Scene | MPS | Total gap ms | GPU contribution ms | Nonkernel contribution ms | Decoder portion ms |
|---|---|---:|---:|---:|---:|
| scene-0770 | False | 1.885 | 1.062 | 0.824 | 0.007 |
| scene-0398 | False | 1.159 | 0.656 | 0.503 | 0.131 |
| scene-0184 | False | 1.620 | -0.041 | 1.660 | 0.910 |
| scene-0245 | False | 2.004 | -0.229 | 2.233 | 1.532 |
| scene-0770 | True | 1.643 | -0.036 | 1.679 | 1.008 |
| scene-0398 | True | 2.068 | 1.368 | 0.700 | 0.284 |
| scene-0184 | True | 1.425 | 0.605 | 0.820 | 0.415 |
| scene-0245 | True | 1.053 | 0.428 | 0.625 | 0.344 |

## Matching retained source frames

| Scene | Common retained frames | Cross-mode latency r |
|---|---:|---:|
| scene-0770 | 222 | 0.302 |
| scene-0398 | 219 | 0.058 |
| scene-0184 | 222 | 0.021 |
| scene-0245 | 220 | 0.265 |

The harness releases unused allocator cache after warmups. Historical startup spikes and their allocation evidence are preserved in the pre-filter archive. Frames outside P1–P99 do not contribute to current correlations or plots; startup_sensitivity.csv records whether the original first input remains in each retained sample.

Primary traces lack CPU sampling, backtraces and CPU context switches. Python/framework dispatch, CPU scheduling and driver launch pacing remain possible mechanisms, not resolved causes. One execution per scene/mode and temporally dependent frames do not establish an image-content effect. P99 estimates remain sparse (<1,000 retained observations); fewer than 100 are especially fragile. First-difference correlations use successive retained observations. Individual-layer attribution would need finer annotations; no additional layer-detail runs were scheduled.

Reproduce after sourcing ROS and the workspace:

```bash
python3 -m closeloop_analyzer.input_data.dino_diagnosis /mmdetection3d_ros2/analysis_outputs/input2
python3 -m closeloop_analyzer.input_data.dino_diagnosis /mmdetection3d_ros2/analysis_outputs/input2 --host-only
python3 -m closeloop_analyzer.input_data.dino_diagnosis /mmdetection3d_ros2/analysis_outputs/input2 --report-only
```
