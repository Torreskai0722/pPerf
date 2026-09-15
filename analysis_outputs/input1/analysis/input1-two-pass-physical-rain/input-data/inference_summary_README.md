# Input1 inference summary

The two requested tables contain 210 rows each: 21 model pairs × 5 conditions × 2 models. All source runs passed the existing campaign validation, and `number_of_frames_executed` counts completed, profiler-matched frames.

- [`mps_summary.csv`](mps_summary.csv): MPS-enabled runs.
- [`non_mps_summary.csv`](non_mps_summary.csv): MPS-disabled runs.
- [`violin_plots/mps`](violin_plots/mps): one MPS plot per model pair.
- [`violin_plots/non_mps`](violin_plots/non_mps): one non-MPS plot per model pair.

Inference time is the validated `inference_e2e_ms` NVTX duration. `inference_time_range_p0_p99_ms` is `p99_ms - p0_ms`; `p0_ms` and `p99_ms` are also retained so the interval endpoints are explicit. Percentiles use NumPy's default linear interpolation. Times are milliseconds and rounded to three decimal places in the tables. Violin distributions are generated with Seaborn's `sns.violinplot`, filter each model-condition group to p1-p99, use blue for LiDAR and orange for image models, and show quartiles as inner lines. The tables retain all frames, including values outside p1-p99.
