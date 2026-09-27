# Preprocessing contention case study

**Inconclusive for causal confirmation.** Exactly twelve executions were completed in the frozen order, with three per policy, MPS disabled, and no replacement runs. Default → Both reduces ViT's preprocessing tail spread and identified-pool runnable-wait P99 in every block, but clock/recording validity gates fail. The affirmative manuscript conclusion is withheld.

| Block | ViT preprocessing P99−P50, Default → Both (ms) | Pool runnable-wait P99, Default → Both (summed thread-ms) |
|---|---:|---:|
| 1 | 14.894 → 3.901 | 69.034 → 3.865 |
| 2 | 19.312 → 4.115 | 84.509 → 4.038 |
| 3 | 15.229 → 2.891 | 88.270 → 2.730 |

Thread limits alone also reduce both ViT endpoints in all three executions. CPU assignment alone reduces ViT's tail spread modestly but raises its median from approximately 6 ms to 14 ms. CenterPoint's CPU-assignment tail grows sharply in blocks 2–3; block 1 has only 99 retained observations, so its tails are withheld. These are policy comparisons: assignment changes OpenCV/PyTorch intra-op defaults from 16/10 to 6/6. No independent factorial effect is claimed.

The first baseline's preprocessing–frame Spearman association is 0.273, versus 0.037 for frame minus preprocessing; its association with summed pool runnable waiting is 0.829. These are within-execution descriptive associations among dependent repeated occurrences. They do not establish an independent effect on the rest of frame processing.

All models have at least 494 matched scheduler switches and 100% reconstructed known-state coverage over retained preprocessing intervals. However, baseline block 1 has a CenterPoint alignment P95 of 54.165 µs and 1,714 lost CPU profiling records; baseline block 2 also reports losses. Every post-warmup graphics readback is 2790 or 2805 MHz, differing from the requested 3105 MHz; memory reads 10501 MHz. Clock resets succeeded after all executions. These failures prevent causal attribution even where the distributions improve.

The predeclared timeline selects ViT occurrence `image:203`, block 1, from the intersection of retained Default/Both occurrences. Its preprocessing is 13.837 versus 0.591 ms. The displayed pool worker is the one with most runnable waiting in that occurrence, with TID tie-breaking. Competitor CSVs identify both model processes, recording helpers, and external activity on eligible CPUs during those intervals. Their finite context window can omit a CPU occupant with no recent switch; these illustrative occupancy exports are not a completeness or causal claim. The complete eligible-CPU switch streams remain in the raw recordings. Recording helpers themselves consume CPU time.

- [Full report](report.md): all execution metrics, matched comparisons, effective settings, quality gates and recording costs.
- [Figure PDF](preprocessing_case.pdf) and [180-dpi PNG](preprocessing_case.png): upper panels pool three separately retained samples per policy, with no second filtering; dark marks show median/IQR. CenterPoint uses a logarithmic axis. The lower timeline is explicitly exploratory.
- [Results](results.json), [comparisons](comparisons.json), [provenance](provenance.json), and [timeline selection](timeline_selection.json): retained/excluded invocation and occurrence identities, source hashes and seeds, execution order, completion counts, drain boundaries, observed pools/masks, and clock evidence.
- [Validation](VALIDATION.md) and [machine-readable checks](validation.json).
- [Reproduction protocol](../../closeloop_perf/studies/preprocessing_case/README.md).

This compact export normalizes absolute workspace paths. Immutable raw artifacts, resolved runtime configurations, bags, full frame/stage/thread exports, failed sanity attempts and superseded reports remain under the original `analysis_outputs/preprocessing_case` artifact root, outside Git. The twelve runs occupy 11.05 GB; lock/run/export wall time is 83.8–89.8 seconds per execution. These are recording costs, not throughput endpoints or estimates of uninstrumented latency overhead. No additional hardware or models were evaluated.
