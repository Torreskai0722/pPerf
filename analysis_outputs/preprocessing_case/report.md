# CenterPoint + ViT preprocessing contention case study

Status: inconclusive; affirmative manuscript conclusion withheld

Fixed synchronized scene-0770:lidar:198 and scene-0770:image:120, chosen from the middle of the cached sequence without latency inspection. The 40 s bag repeats these sources at 20/12 Hz with distinct occurrence IDs. Five warmup inferences per model; ViT resize 512×512; launch offsets 0/1 s; replay waits for readiness. MPS disabled; requested GPU clock locks 3105/10501 MHz (readbacks below). Replay/relay placement is held at CPUs 12–15. Baseline means default model CPU settings.

CPU preprocessing is the existing preprocess range and its constituent transforms. Frame latency spans callback entry to CUDA-confirmed completion, including decoding. Delivery delay and internal data_preprocessor ranges are separate. GPU voxelization and CUDA synchronization are not CPU runnable waiting.

P1–P99 inclusive linear cutoffs are computed once per execution/model/actual scene from original completed non-warmup inference latencies. All component statistics and correlations use those retained invocation identities. Repeated source payloads are separate occurrences. Full frame/stage exports and retained IDs are preserved alongside every per-run result. The figure pools the three separately filtered executions without re-filtering. Figure summaries show median and interquartile range. Tails are withheld for any execution/model or matched subset with fewer than 100 retained observations; pooling does not repair that gate. The lower panel uses block 1 ViT, baseline occurrence nearest its retained preprocessing P95 among occurrences also retained in Both.

| Block | Condition | Model | Retained | P50 (ms) | P99−P50 (ms) | P99−P1 (ms) | Pool wait P99 (thread-ms) | ρ(pre,frame) | ρ(pre,frame−pre) | ρ(pre,pool wait) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | default | vit-upernet | 470 | 6.013 | 14.894 | 20.333 | 69.034 | 0.273 | 0.037 | 0.829 |
| 1 | default | centerpoint | 149 | 22.262 | 65.843 | 81.471 | 145.540 | 0.387 | -0.259 | 0.965 |
| 1 | cpu | vit-upernet | 470 | 14.151 | 13.995 | 27.521 | 39.964 | 0.451 | 0.106 | 0.820 |
| 1 | cpu | centerpoint | 99 | 29.469 | NA | NA | NA | 0.863 | -0.212 | 0.970 |
| 1 | threads | vit-upernet | 470 | 0.635 | 3.417 | 3.507 | 3.317 | 0.197 | 0.138 | 0.611 |
| 1 | threads | centerpoint | 157 | 10.172 | 5.601 | 7.721 | 3.172 | -0.214 | -0.705 | 0.151 |
| 1 | both | vit-upernet | 470 | 0.673 | 3.901 | 3.995 | 3.865 | 0.158 | 0.074 | 0.337 |
| 1 | both | centerpoint | 157 | 12.920 | 2.882 | 7.140 | 2.385 | -0.091 | -0.373 | 0.163 |
| 2 | threads | vit-upernet | 470 | 0.633 | 0.580 | 0.662 | 0.557 | 0.220 | 0.194 | 0.517 |
| 2 | threads | centerpoint | 157 | 9.132 | 5.114 | 6.355 | 1.726 | -0.154 | -0.673 | 0.404 |
| 2 | both | vit-upernet | 470 | 0.675 | 4.115 | 4.213 | 4.038 | 0.088 | 0.003 | 0.401 |
| 2 | both | centerpoint | 157 | 12.909 | 4.462 | 9.334 | 4.181 | -0.189 | -0.547 | 0.221 |
| 2 | default | vit-upernet | 470 | 6.026 | 19.312 | 24.741 | 84.509 | 0.320 | 0.098 | 0.731 |
| 2 | default | centerpoint | 149 | 22.987 | 50.419 | 66.961 | 130.073 | 0.265 | -0.368 | 0.956 |
| 2 | cpu | vit-upernet | 470 | 14.321 | 12.776 | 26.460 | 43.298 | 0.493 | 0.170 | 0.824 |
| 2 | cpu | centerpoint | 100 | 18.729 | 1569.965 | 1580.187 | 1588.675 | 0.887 | -0.140 | 0.946 |
| 3 | both | vit-upernet | 470 | 0.666 | 2.891 | 2.981 | 2.730 | 0.153 | 0.087 | 0.264 |
| 3 | both | centerpoint | 157 | 12.890 | 2.913 | 3.761 | 0.901 | -0.039 | -0.479 | 0.301 |
| 3 | threads | vit-upernet | 470 | 0.641 | 4.275 | 4.369 | 4.314 | 0.215 | 0.161 | 0.611 |
| 3 | threads | centerpoint | 157 | 9.960 | 3.913 | 5.873 | 1.360 | -0.106 | -0.564 | 0.317 |
| 3 | cpu | vit-upernet | 470 | 14.602 | 13.729 | 27.689 | 42.727 | 0.490 | 0.233 | 0.768 |
| 3 | cpu | centerpoint | 120 | 13.493 | 948.490 | 953.749 | 955.188 | 0.871 | -0.220 | 0.942 |
| 3 | default | vit-upernet | 470 | 6.046 | 15.229 | 20.665 | 88.270 | 0.344 | 0.118 | 0.788 |
| 3 | default | centerpoint | 148 | 25.835 | 60.699 | 79.728 | 126.675 | 0.426 | -0.264 | 0.956 |

These are configuration-policy comparisons: affinity can itself change native defaults, so CPU placement and library parallelism are not independent factorial effects. Spearman associations are within execution and descriptive; preprocessing is part of frame latency, and the frame-minus-preprocessing diagnostic is reported explicitly. Repeated frames are not statistically independent observations. Results apply only to this recorded workload and hardware.

Recording cost includes wall duration, artifact bytes, snapshot time, and offline analysis time in each result. Relative timing perturbation was not estimated because the protocol has no uninstrumented execution; instrumentation is identical across policies.

CPU-sharing interference context: [Elmougy et al., Diagnosing the Interference on CPU-GPU Synchronization Caused by CPU Sharing in Multi-Tenant GPU Clouds (2021)](https://doi.org/10.1109/IPCCC51483.2021.9679439). Library-pool context: [PyTorch, Optimizing LibTorch-based inference engine memory usage and thread-pooling](https://pytorch.org/blog/optimizing-libtorch/). Their cloud/LibTorch settings differ from this ROS/OpenMMLab fixed-input GPU workload.

Recording failures: []

## Recording quality and effective settings

| Run | Causal attribution allowed | Minimum matched switches/model | Maximum alignment P95 (µs) | Minimum known state (%) |
|---|---|---:|---:|---:|
| preprocessing-b1-default | False | 764 | 54.165 | 100.000 |
| preprocessing-b1-cpu | False | 648 | 4.327 | 100.000 |
| preprocessing-b1-threads | False | 2000 | 4.688 | 100.000 |
| preprocessing-b1-both | False | 1999 | 4.985 | 100.000 |
| preprocessing-b2-threads | False | 1999 | 4.436 | 100.000 |
| preprocessing-b2-both | False | 2000 | 4.727 | 100.000 |
| preprocessing-b2-default | False | 643 | 33.055 | 100.000 |
| preprocessing-b2-cpu | False | 1971 | 26.666 | 100.000 |
| preprocessing-b3-both | False | 1999 | 4.571 | 100.000 |
| preprocessing-b3-threads | False | 1999 | 4.990 | 100.000 |
| preprocessing-b3-cpu | False | 1976 | 27.781 | 100.000 |
| preprocessing-b3-default | False | 494 | 4.382 | 100.000 |

preprocessing-b1-default: Nsight/BPF reported trace loss; causal attribution rejected; scheduler clock alignment failed; attribution is exploratory only; vit-upernet GPU clock readback differs: 2805, 10501; centerpoint GPU clock readback differs: 2790, 10501.

preprocessing-b1-cpu: vit-upernet GPU clock readback differs: 2790, 10501; fewer than 100 retained observations; tail statistics withheld: centerpoint; centerpoint GPU clock readback differs: 2805, 10501.

preprocessing-b1-threads: vit-upernet GPU clock readback differs: 2790, 10501; centerpoint GPU clock readback differs: 2790, 10501.

preprocessing-b1-both: vit-upernet GPU clock readback differs: 2790, 10501; centerpoint GPU clock readback differs: 2790, 10501.

preprocessing-b2-threads: vit-upernet GPU clock readback differs: 2790, 10501; centerpoint GPU clock readback differs: 2790, 10501.

preprocessing-b2-both: vit-upernet GPU clock readback differs: 2805, 10501; centerpoint GPU clock readback differs: 2790, 10501.

preprocessing-b2-default: Nsight/BPF reported trace loss; causal attribution rejected; vit-upernet GPU clock readback differs: 2790, 10501; centerpoint GPU clock readback differs: 2790, 10501.

preprocessing-b2-cpu: vit-upernet GPU clock readback differs: 2790, 10501; centerpoint GPU clock readback differs: 2805, 10501.

preprocessing-b3-both: vit-upernet GPU clock readback differs: 2790, 10501; centerpoint GPU clock readback differs: 2790, 10501.

preprocessing-b3-threads: vit-upernet GPU clock readback differs: 2790, 10501; centerpoint GPU clock readback differs: 2790, 10501.

preprocessing-b3-cpu: vit-upernet GPU clock readback differs: 2790, 10501; centerpoint GPU clock readback differs: 2790, 10501.

preprocessing-b3-default: vit-upernet GPU clock readback differs: 2805, 10501; centerpoint GPU clock readback differs: 2790, 10501.

| Run | Model | Completed / retained | OpenCV / Torch intra / inter | Observed graphics / memory (MHz) |
|---|---|---:|---:|---:|
| preprocessing-b1-default | vit-upernet | 480 / 470 | 16 / 10 / 10 | 2805, 10501 |
| preprocessing-b1-default | centerpoint | 153 / 149 | 16 / 10 / 10 | 2790, 10501 |
| preprocessing-b1-cpu | vit-upernet | 480 / 470 | 6 / 6 / 10 | 2790, 10501 |
| preprocessing-b1-cpu | centerpoint | 103 / 99 | 6 / 6 / 10 | 2805, 10501 |
| preprocessing-b1-threads | vit-upernet | 480 / 470 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b1-threads | centerpoint | 161 / 157 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b1-both | vit-upernet | 480 / 470 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b1-both | centerpoint | 161 / 157 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b2-threads | vit-upernet | 480 / 470 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b2-threads | centerpoint | 161 / 157 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b2-both | vit-upernet | 480 / 470 | 3 / 3 / 1 | 2805, 10501 |
| preprocessing-b2-both | centerpoint | 161 / 157 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b2-default | vit-upernet | 480 / 470 | 16 / 10 / 10 | 2790, 10501 |
| preprocessing-b2-default | centerpoint | 153 / 149 | 16 / 10 / 10 | 2790, 10501 |
| preprocessing-b2-cpu | vit-upernet | 480 / 470 | 6 / 6 / 10 | 2790, 10501 |
| preprocessing-b2-cpu | centerpoint | 104 / 100 | 6 / 6 / 10 | 2805, 10501 |
| preprocessing-b3-both | vit-upernet | 480 / 470 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b3-both | centerpoint | 161 / 157 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b3-threads | vit-upernet | 480 / 470 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b3-threads | centerpoint | 161 / 157 | 3 / 3 / 1 | 2790, 10501 |
| preprocessing-b3-cpu | vit-upernet | 480 / 470 | 6 / 6 / 10 | 2790, 10501 |
| preprocessing-b3-cpu | centerpoint | 124 / 120 | 6 / 6 / 10 | 2790, 10501 |
| preprocessing-b3-default | vit-upernet | 480 / 470 | 16 / 10 / 10 | 2805, 10501 |
| preprocessing-b3-default | centerpoint | 152 / 148 | 16 / 10 / 10 | 2790, 10501 |

Native capacities and every per-thread CPU mask are retained in each result. Nsight helper threads are identified separately and retain the inherited CPU mask. Reported pool times below sum identified worker-thread intervals; they are not elapsed wall time. Every derived row references the retained invocation and occurrence lists in its execution result.

| Run | Model | Caller running / runnable waiting / blocked P99 (ms) | Pool running / runnable waiting / blocked P99 (thread-ms) |
|---|---|---:|---:|
| preprocessing-b1-default | vit-upernet | 14.790 / 15.526 / 2.613 | 111.242 / 69.034 / 410.845 |
| preprocessing-b1-default | centerpoint | 88.096 / 14.749 / 1.398 | 652.977 / 145.540 / 283.636 |
| preprocessing-b1-cpu | vit-upernet | 17.798 / 15.965 / 0.577 | 90.451 / 39.964 / 236.214 |
| preprocessing-b1-cpu | centerpoint | NA / NA / NA | NA / NA / NA |
| preprocessing-b1-threads | vit-upernet | 4.052 / 0.000 / 0.000 | 4.543 / 3.317 / 28.609 |
| preprocessing-b1-threads | centerpoint | 15.773 / 0.020 / 0.000 | 28.048 / 3.172 / 79.072 |
| preprocessing-b1-both | vit-upernet | 4.574 / 0.000 / 0.000 | 5.053 / 3.865 / 32.173 |
| preprocessing-b1-both | centerpoint | 15.801 / 0.024 / 0.022 | 31.101 / 2.385 / 79.224 |
| preprocessing-b2-threads | vit-upernet | 1.213 / 0.000 / 0.000 | 1.596 / 0.557 / 8.739 |
| preprocessing-b2-threads | centerpoint | 14.246 / 0.022 / 0.000 | 26.254 / 1.726 / 71.410 |
| preprocessing-b2-both | vit-upernet | 4.790 / 0.000 / 0.000 | 5.280 / 4.038 / 33.750 |
| preprocessing-b2-both | centerpoint | 17.372 / 0.023 / 0.000 | 31.152 / 4.181 / 87.056 |
| preprocessing-b2-default | vit-upernet | 14.226 / 15.142 / 2.684 | 128.439 / 84.509 / 474.585 |
| preprocessing-b2-default | centerpoint | 73.407 / 12.909 / 1.159 | 528.901 / 130.073 / 233.625 |
| preprocessing-b2-cpu | vit-upernet | 18.379 / 15.999 / 2.051 | 99.230 / 43.298 / 223.018 |
| preprocessing-b2-cpu | centerpoint | 1588.659 / 136.909 / 0.126 | 6352.349 / 1588.675 / 11123.307 |
| preprocessing-b3-both | vit-upernet | 3.557 / 0.000 / 0.000 | 3.987 / 2.730 / 25.110 |
| preprocessing-b3-both | centerpoint | 15.792 / 0.023 / 0.009 | 31.343 / 0.901 / 79.257 |
| preprocessing-b3-threads | vit-upernet | 4.911 / 0.000 / 0.000 | 5.265 / 4.314 / 34.597 |
| preprocessing-b3-threads | centerpoint | 13.874 / 0.014 / 0.006 | 27.093 / 1.360 / 69.592 |
| preprocessing-b3-cpu | vit-upernet | 18.194 / 16.001 / 1.515 | 121.591 / 42.727 / 228.558 |
| preprocessing-b3-cpu | centerpoint | 961.972 / 20.325 / 0.000 | 3852.356 / 955.188 / 6736.259 |
| preprocessing-b3-default | vit-upernet | 15.622 / 13.706 / 5.860 | 104.210 / 88.270 / 407.430 |
| preprocessing-b3-default | centerpoint | 86.286 / 15.236 / 1.612 | 630.303 / 126.675 / 293.690 |

## Matched occurrences

Differences are intervention minus Default within block, after intersecting the two retained occurrence sets; no re-filtering. Pooled plots do not replace these comparisons. `comparisons.json` also contains P99−P50 and pool-wait P99 on each matched subset.

| Block | Policy | Model | Matched | Median Δ preprocessing / frame / frame−preprocessing (ms) | Full retained Δ spread / pool-wait P99 (ms) |
|---|---|---|---:|---:|---:|
| 1 | cpu | vit-upernet | 460 | 8.356 / 2.469 / -3.197 | -0.899 / -29.070 |
| 1 | cpu | centerpoint | 20 | -7.182 / -18.730 / -5.656 | NA / NA |
| 1 | threads | vit-upernet | 461 | -5.339 / -5.072 / -0.141 | -11.478 / -65.716 |
| 1 | threads | centerpoint | 40 | -15.302 / -5.752 / -2.788 | -60.242 / -142.368 |
| 1 | both | vit-upernet | 461 | -5.299 / -5.018 / -0.288 | -10.993 / -65.169 |
| 1 | both | centerpoint | 39 | -12.437 / -6.183 / -4.969 | -62.962 / -143.155 |
| 2 | cpu | vit-upernet | 460 | 8.422 / 3.248 / -3.421 | -6.535 / -41.211 |
| 2 | cpu | centerpoint | 16 | 56.951 / 77.514 / 17.762 | 1519.546 / 1458.602 |
| 2 | threads | vit-upernet | 460 | -5.373 / -5.125 / -0.286 | -18.731 / -83.952 |
| 2 | threads | centerpoint | 62 | -17.286 / -4.793 / 3.419 | -45.306 / -128.346 |
| 2 | both | vit-upernet | 461 | -5.342 / -5.233 / -0.252 | -15.197 / -80.472 |
| 2 | both | centerpoint | 62 | -12.304 / -6.047 / -1.858 | -45.957 / -125.892 |
| 3 | cpu | vit-upernet | 460 | 8.613 / 6.495 / -1.379 | -1.500 / -45.543 |
| 3 | cpu | centerpoint | 27 | 0.006 / 2.900 / 2.894 | 887.791 / 828.513 |
| 3 | threads | vit-upernet | 461 | -5.387 / -4.720 / -0.090 | -10.953 / -83.956 |
| 3 | threads | centerpoint | 56 | -14.348 / -7.604 / -0.177 | -56.786 / -125.315 |
| 3 | both | vit-upernet | 462 | -5.356 / -6.075 / -0.660 | -12.338 / -85.541 |
| 3 | both | centerpoint | 22 | -16.575 / -20.486 / -10.301 | -57.786 / -125.774 |

## Recording cost

| Run | Lock/run/export wall time (s) | Raw artifact size (MB) | Offline analysis (s) |
|---|---:|---:|---:|
| preprocessing-b1-default | 89.78 | 980.24 | 27.38 |
| preprocessing-b1-cpu | 88.84 | 915.24 | 25.12 |
| preprocessing-b1-threads | 84.97 | 914.74 | 24.23 |
| preprocessing-b1-both | 84.74 | 910.79 | 24.68 |
| preprocessing-b2-threads | 84.11 | 899.94 | 23.95 |
| preprocessing-b2-both | 86.01 | 919.01 | 23.99 |
| preprocessing-b2-default | 87.86 | 955.56 | 26.45 |
| preprocessing-b2-cpu | 86.87 | 897.07 | 23.42 |
| preprocessing-b3-both | 84.48 | 905.78 | 23.32 |
| preprocessing-b3-threads | 83.80 | 897.75 | 22.90 |
| preprocessing-b3-cpu | 85.95 | 889.81 | 24.03 |
| preprocessing-b3-default | 88.27 | 962.57 | 26.16 |

These costs include initialization and export and are not estimates of latency overhead. The collector and its helper threads may themselves compete for CPU time. Raw BPF switches, Nsight diagnostics, clock-lock/reset responses, completion records and drain boundaries remain immutable under `runs/`. The preliminary baseline diagnosis was saved before the first intervention and is retained independently of this final report. Reproduction commands and the frozen order are in `closeloop_perf/studies/preprocessing_case/README.md`.
