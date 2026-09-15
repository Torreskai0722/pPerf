# YOLOv3 Pair-Dependent Inference-Time Variation

## Summary

YOLOv3 showed little direct sensitivity to weather-corrupted inputs when run alone, but its p0–p99 inference-time range responded differently when it shared the GPU with different LiDAR models. This indicates that the observed weather response is not solely a property of YOLOv3: weather can change the co-runner's workload, which changes the contention experienced by YOLOv3.

This effect is best described as **co-runner-mediated, input-induced interference**. It is also a model-selection result because the model pair and execution mode affect both YOLOv3's baseline variability and the direction of its response to rain.

Relative change is calculated against the clear run for the same model pair and MPS mode:

`relative change = (rain p0–p99 range − clear p0–p99 range) / clear p0–p99 range × 100%`

Positive values indicate amplification of YOLOv3's inference-time range; negative values indicate damping.

## Non-MPS results

| Co-running LiDAR model | Rain rate | Clear range (ms) | Rain range (ms) | Absolute change (ms) | Relative change |
|---|---:|---:|---:|---:|---:|
| CenterPoint | 7.5 mm/h | 41.721 | 41.897 | +0.176 | +0.42% |
| CenterPoint | 15 mm/h | 41.721 | 41.316 | −0.405 | −0.97% |
| CenterPoint | 25 mm/h | 41.721 | 41.982 | +0.261 | +0.63% |
| CenterPoint | 50 mm/h | 41.721 | 41.133 | −0.588 | −1.41% |
| PointPillars | 7.5 mm/h | 18.387 | 14.797 | −3.590 | −19.53% |
| PointPillars | 15 mm/h | 18.387 | 15.386 | −3.001 | −16.32% |
| PointPillars | 25 mm/h | 18.387 | 15.706 | −2.681 | −14.58% |
| PointPillars | 50 mm/h | 18.387 | 17.134 | −1.253 | −6.82% |
| 3DSSD | 7.5 mm/h | 36.406 | 36.681 | +0.276 | +0.76% |
| 3DSSD | 15 mm/h | 36.406 | 36.766 | +0.360 | +0.99% |
| 3DSSD | 25 mm/h | 36.406 | 36.293 | −0.113 | −0.31% |
| 3DSSD | 50 mm/h | 36.406 | 37.857 | +1.451 | +3.99% |

The non-MPS data most directly supports the original observation:

- With CenterPoint, YOLOv3 was nearly insensitive to rain; every range change remained within ±1.41%.
- With PointPillars, YOLOv3's range decreased at every rain rate, with reductions of 6.82%–19.53%.
- With 3DSSD, the range generally increased slightly. The largest increase was 3.99% at 50 mm/h, while 25 mm/h produced a small 0.31% reduction.

## MPS results

| Co-running LiDAR model | Rain rate | Clear range (ms) | Rain range (ms) | Absolute change (ms) | Relative change |
|---|---:|---:|---:|---:|---:|
| CenterPoint | 7.5 mm/h | 47.234 | 44.288 | −2.946 | −6.24% |
| CenterPoint | 15 mm/h | 47.234 | 44.572 | −2.662 | −5.64% |
| CenterPoint | 25 mm/h | 47.234 | 42.888 | −4.346 | −9.20% |
| CenterPoint | 50 mm/h | 47.234 | 43.669 | −3.565 | −7.55% |
| PointPillars | 7.5 mm/h | 12.529 | 11.653 | −0.876 | −6.99% |
| PointPillars | 15 mm/h | 12.529 | 11.443 | −1.086 | −8.66% |
| PointPillars | 25 mm/h | 12.529 | 12.597 | +0.068 | +0.54% |
| PointPillars | 50 mm/h | 12.529 | 12.013 | −0.516 | −4.12% |
| 3DSSD | 7.5 mm/h | 7.226 | 7.585 | +0.359 | +4.96% |
| 3DSSD | 15 mm/h | 7.226 | 7.970 | +0.743 | +10.29% |
| 3DSSD | 25 mm/h | 7.226 | 7.121 | −0.106 | −1.46% |
| 3DSSD | 50 mm/h | 7.226 | 8.052 | +0.825 | +11.42% |

Under MPS, pair dependence remained but the response changed:

- CenterPoint consistently dampened YOLOv3's range by 5.64%–9.20%.
- PointPillars usually dampened the range by 4.12%–8.66%, except for a negligible 0.54% increase at 25 mm/h.
- 3DSSD generally amplified the range by 4.96%–11.42%, except for a 1.46% reduction at 25 mm/h. Because the clear range was only 7.226 ms, these relatively large percentages correspond to absolute changes below 0.83 ms.

## Interpretation

The same YOLOv3 model did not have a uniform response to a given weather condition. Its response depended on the computation and resource-use pattern of the co-running LiDAR model. The clearest contrast is between PointPillars and 3DSSD: PointPillars generally reduced YOLOv3's range, whereas 3DSSD generally increased it. CenterPoint was almost neutral without MPS but produced consistent damping with MPS.

This supports an interaction pathway:

`weather-altered input → co-runner workload change → altered GPU contention → YOLOv3 latency-distribution change`

The result should not be interpreted as direct evidence that rain substantially changes YOLOv3's own computation. Instead, the limited isolated-model response and the divergent paired-model responses indicate that much of the observed change is mediated through the co-running model. Accordingly, inference-time predictability should be evaluated per model pair and execution mode rather than inferred from isolated model behavior alone.

The results are descriptive: the experiment used one execution per condition, so the differences should not be presented as statistically confirmed causal effects. The comparison also concerns p0–p99 range, which measures distribution width and tail behavior rather than a change in mean inference time.

## Data sources

- `weather_comparisons_mps_off.csv`
- `weather_comparisons_mps_on.csv`
- `non_mps_summary.csv`
- `mps_summary.csv`
