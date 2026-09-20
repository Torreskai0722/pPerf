# CenterPoint isolated inference diagnosis

The strongest measured module association in every scene and both MPS modes is GPU hard voxelization inside `data_preprocessor`. This is inside the CUDA-completed inference boundary; the separately logged CPU preprocessing and message decoding are outside it.

All plots and metrics use each execution's P1–P99-filtered inference frames. Cutoffs and exclusions are recorded in summary.json; P50/P99 are recomputed on retained frames. Components share the same frame mask. GPU module durations use existing correlated launch attribution. Included kernels lie inside their inference boundaries and do not overlap, allowing additive accounting. Difference correlations use successive retained observations.

| Scene | MPS | Frames | P50 ms | P99-P50 ms | Voxelization mean ms | Pearson r | Consecutive-difference r |
|---|---|---:|---:|---:|---:|---:|---:|
| scene-0770 | off | 143 | 119.813 | 4.609 | 74.182 | 0.891 | 0.909 |
| scene-0398 | off | 142 | 120.088 | 5.992 | 74.363 | 0.960 | 0.854 |
| scene-0184 | off | 138 | 125.465 | 5.502 | 78.822 | 0.963 | 0.905 |
| scene-0245 | off | 184 | 92.529 | 13.950 | 48.836 | 0.980 | 0.782 |
| scene-0770 | on | 142 | 120.048 | 4.352 | 74.105 | 0.872 | 0.878 |
| scene-0398 | on | 142 | 119.916 | 4.967 | 74.309 | 0.975 | 0.784 |
| scene-0184 | on | 137 | 125.259 | 5.965 | 78.773 | 0.947 | 0.781 |
| scene-0245 | on | 182 | 92.578 | 12.352 | 48.944 | 0.989 | 0.939 |

## Mechanism and scope

The retained configuration uses deterministic hard voxelization. Two measured kernels dominate this module: `point_to_voxelidx_kernel` scans preceding points for matching voxel coordinates, stopping when its point-per-voxel limit is met; `determin_voxel_num` walks the points sequentially and is launched with one block and one thread. Input size, spatial occupancy and source order can therefore affect their work. Source implementation hashes and resolved CPU pipeline settings are retained in `input_geometry_provenance.json`.

The actual inferencer pipeline pads nine missing sweeps with copies of the current cloud (removing close points from the added copies), then applies the spatial range filter. `input_geometry.csv` reconstructs this CPU pipeline for every unique timed source frame. These are reconstructed inputs/logical scan iterations, not recorded GPU voxel counts or hardware cycles. Raw message point counts alone are not the voxelizer's workload.

| Scene | MPS | Mean voxel input points | Mean occupied voxels | Logical point-scan work vs inference r |
|---|---|---:|---:|---:|
| scene-0770 | off | 259346 | 17701 | 0.203 |
| scene-0398 | off | 263751 | 17055 | 0.848 |
| scene-0184 | off | 272279 | 18294 | 0.778 |
| scene-0245 | off | 198384 | 11391 | 0.942 |
| scene-0770 | on | 259346 | 17702 | 0.247 |
| scene-0398 | on | 263747 | 17036 | 0.868 |
| scene-0184 | on | 272388 | 18299 | 0.812 |
| scene-0245 | on | 198398 | 11393 | 0.926 |

Input geometry explains substantially more variation in scenes 0398, 0184 and 0245 than in 0770. In 0770, voxelization still tracks inference closely while reconstructed work is relatively stable; input geometry alone does not explain its runtime variation. Kernel timing fluctuations on identical source frames also remain between modes.

## Matching actual source frames between modes

| Scene | Common retained frames | Filtered-sample gap change ms | Common-frame gap change ms | Correlation of voxelization and total per-frame mode changes |
|---|---:|---:|---:|---:|
| scene-0770 | 99 | -0.257 | -0.784 | 0.829 |
| scene-0398 | 92 | -1.025 | -1.701 | 0.758 |
| scene-0184 | 41 | 0.462 | -3.583 | 0.904 |
| scene-0245 | 100 | -1.598 | -1.997 | 0.770 |

## Accounting for the observed tail-gap change

Entries are MPS-on minus MPS-off, in milliseconds. For each run, each component is evaluated on the same interpolated frames defining the total P50 and P99. The three component columns add to the total change, but this is descriptive endpoint accounting, not an intervention or a module's own percentile gap.

| Scene | Total gap change | Voxelization contribution change | Other kernels contribution change | Nonkernel contribution change |
|---|---:|---:|---:|---:|
| scene-0770 | -0.257 | -2.243 | 0.102 | 1.884 |
| scene-0398 | -1.025 | -1.531 | 0.110 | 0.396 |
| scene-0184 | 0.462 | -0.306 | 0.579 | 0.189 |
| scene-0245 | -1.598 | -0.068 | -0.019 | -1.512 |

Each condition has one execution and temporally dependent observations. All full-sample P99 estimates have fewer than 1,000 observations; common-frame subsets below 100 are especially fragile. Mode differences remain observational because execution order, runtime state and frame timing are not independently replicated. A module's contribution is part of total latency, so high correlation alone is not causal proof.

The input-driven voxelization mechanism is supported by implementation inspection, reconstructed workload and recorded kernel timings. The precise cause of occasional stalls and the smaller MPS-versus-disabled tail difference remains unresolved. No finer neural-network layer labels are inferred from the top-level annotations. Resolving the residual mechanism would require targeted repeated timing and launch/memory/scheduler evidence; no additional GPU or layer-detail runs were scheduled.

`associations.csv` also reports additive changes at the total latency's P50/P99 endpoint frames. They sum to total P99-P50, but are not each module's independent P99-P50 and can be negative. `nonkernel_ms` includes memory copies, host work, launch gaps and synchronization, not exclusively CPU execution.

Reproduce after sourcing ROS and the workspace:

```bash
python3 -m closeloop_analyzer.input_data.centerpoint_diagnosis /mmdetection3d_ros2/analysis_outputs/input2
python3 -m closeloop_profiler.centerpoint_input_geometry /mmdetection3d_ros2/analysis_outputs/input2
python3 -m closeloop_analyzer.input_data.centerpoint_diagnosis /mmdetection3d_ros2/analysis_outputs/input2 --report-only
```
