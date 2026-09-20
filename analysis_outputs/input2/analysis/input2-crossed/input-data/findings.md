# Input2-crossed findings — P1–P99 filtered

All current latency statistics and plots use the same retained per-execution/model frames. P50/P99 are recomputed after filtering; original cutoffs and exclusions are retained. Full evidence and the historical selection remain unchanged. See [complete tables and contrasts](report.md), [baseline violins](isolated_baselines.md), [CenterPoint diagnosis](centerpoint_diagnosis/report.md), and [DINO diagnosis](dino_diagnosis/report.md).

The practical predictability threshold is an absolute change of approximately 1 ms in P99−P50. The table uses a strict ≤1 ms comparison. Isolated mode comparisons have one execution per condition and are observations, not statistical equivalence tests. 3DSSD and YOLOv3 are assessed separately.

| Model | Scene | MPS-off P99−P50 ms | MPS-on P99−P50 ms | On−off ms | Within 1 ms |
|---|---|---:|---:|---:|---|
| 3dssd | scene-0184 | 0.871 | 0.646 | -0.225 | True |
| 3dssd | scene-0245 | 0.676 | 0.682 | +0.006 | True |
| 3dssd | scene-0398 | 0.811 | 0.587 | -0.224 | True |
| 3dssd | scene-0770 | 1.071 | 0.781 | -0.289 | True |
| centerpoint | scene-0184 | 5.502 | 5.965 | +0.462 | True |
| centerpoint | scene-0245 | 13.950 | 12.352 | -1.598 | False |
| centerpoint | scene-0398 | 5.992 | 4.967 | -1.025 | False |
| centerpoint | scene-0770 | 4.609 | 4.352 | -0.257 | True |
| dino | scene-0184 | 1.620 | 1.425 | -0.195 | True |
| dino | scene-0245 | 2.004 | 1.053 | -0.951 | True |
| dino | scene-0398 | 1.159 | 2.068 | +0.909 | True |
| dino | scene-0770 | 1.885 | 1.643 | -0.243 | True |
| mask-rcnn | scene-0184 | 2.926 | 2.972 | +0.046 | True |
| mask-rcnn | scene-0245 | 3.406 | 3.121 | -0.285 | True |
| mask-rcnn | scene-0398 | 3.875 | 3.906 | +0.031 | True |
| mask-rcnn | scene-0770 | 2.973 | 2.636 | -0.336 | True |
| pointpillars | scene-0184 | 1.895 | 2.019 | +0.125 | True |
| pointpillars | scene-0245 | 2.398 | 2.483 | +0.085 | True |
| pointpillars | scene-0398 | 2.101 | 2.238 | +0.137 | True |
| pointpillars | scene-0770 | 1.867 | 1.223 | -0.644 | True |
| vit-upernet | scene-0184 | 0.954 | 0.658 | -0.297 | True |
| vit-upernet | scene-0245 | 0.852 | 1.971 | +1.118 | False |
| vit-upernet | scene-0398 | 0.666 | 0.747 | +0.081 | True |
| vit-upernet | scene-0770 | 0.697 | 0.851 | +0.154 | True |
| yolov3 | scene-0184 | 2.809 | 3.560 | +0.752 | True |
| yolov3 | scene-0245 | 4.294 | 2.301 | -1.993 | False |
| yolov3 | scene-0398 | 2.820 | 2.910 | +0.091 | True |
| yolov3 | scene-0770 | 3.130 | 3.405 | +0.274 | True |

Crossed contrasts compare actual executed input combinations at fixed co-runner or own input. Their execution-level means, standard deviations and ranges preserve independent repetitions. Common-frame comparisons intersect the retained source identities. Completion coverage and observed throughput remain evidence about the full run; filtered throughput uses retained count over the original elapsed duration. Mechanism correlations are descriptive and do not establish causation.

Historical selection.json identifies the actual completed matrix. Revised P1–P99 screening choices are in selection_p1_p99.json; no confirmation runs are relabeled to match revised choices. Filtered samples remain temporally dependent; P99 with fewer than 100 retained observations is especially fragile and fewer than 1,000 is sparse-tail. No additional runs were scheduled.
