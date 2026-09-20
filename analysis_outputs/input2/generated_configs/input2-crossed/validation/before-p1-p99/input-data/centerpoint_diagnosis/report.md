# CenterPoint isolated inference diagnosis

The strongest measured module association in every scene and both MPS modes is GPU hard voxelization inside `data_preprocessor`. This is inside the CUDA-completed inference boundary; the separately logged CPU preprocessing and message decoding are outside it.

Correlations below are per execution, across completed non-warmup source frames. No percentile trimming is applied. GPU module durations use existing correlated launch attribution. All included kernels lie inside their inference boundaries and do not overlap, allowing additive duration accounting.

| Scene | MPS | Frames | P50 ms | P99-P50 ms | Voxelization mean ms | Pearson r | Consecutive-difference r |
|---|---|---:|---:|---:|---:|---:|---:|
| scene-0770 | off | 147 | 119.813 | 8.099 | 74.298 | 0.944 | 0.955 |
| scene-0398 | off | 146 | 120.088 | 8.211 | 74.363 | 0.967 | 0.909 |
| scene-0184 | off | 142 | 125.465 | 11.043 | 78.910 | 0.968 | 0.945 |
| scene-0245 | off | 188 | 92.529 | 14.294 | 48.883 | 0.983 | 0.784 |
| scene-0770 | on | 146 | 120.048 | 7.736 | 74.223 | 0.944 | 0.945 |
| scene-0398 | on | 146 | 119.916 | 6.133 | 74.305 | 0.977 | 0.854 |
| scene-0184 | on | 141 | 125.259 | 8.002 | 78.819 | 0.956 | 0.906 |
| scene-0245 | on | 186 | 92.578 | 12.875 | 48.987 | 0.990 | 0.937 |

## Mechanism and scope

The retained configuration uses deterministic hard voxelization. Two measured kernels dominate this module: `point_to_voxelidx_kernel` scans preceding points for matching voxel coordinates, stopping when its point-per-voxel limit is met; `determin_voxel_num` walks the points sequentially and is launched with one block and one thread. Input size, spatial occupancy and source order can therefore affect their work. Source implementation hashes and resolved CPU pipeline settings are retained in `input_geometry_provenance.json`.

The actual inferencer pipeline pads nine missing sweeps with copies of the current cloud (removing close points from the added copies), then applies the spatial range filter. `input_geometry.csv` reconstructs this CPU pipeline for every unique timed source frame. These are reconstructed inputs/logical scan iterations, not recorded GPU voxel counts or hardware cycles. Raw message point counts alone are not the voxelizer's workload.

| Scene | MPS | Mean voxel input points | Mean occupied voxels | Logical point-scan work vs inference r |
|---|---|---:|---:|---:|
| scene-0770 | off | 259348 | 17702 | 0.197 |
| scene-0398 | off | 263806 | 16980 | 0.843 |
| scene-0184 | off | 272275 | 18277 | 0.757 |
| scene-0245 | off | 198487 | 11424 | 0.948 |
| scene-0770 | on | 259335 | 17701 | 0.185 |
| scene-0398 | on | 263800 | 16977 | 0.862 |
| scene-0184 | on | 272208 | 18286 | 0.749 |
| scene-0245 | on | 198494 | 11427 | 0.934 |

Input geometry explains substantially more variation in scenes 0398, 0184 and 0245 than in 0770. In 0770, voxelization still tracks inference closely while reconstructed work is relatively stable; input geometry alone does not explain its runtime variation. Kernel timing fluctuations on identical source frames also remain between modes.

## Matching actual source frames between modes

| Scene | Common frames | Full-sample gap change ms | Common-frame gap change ms | Correlation of voxelization and total per-frame mode changes |
|---|---:|---:|---:|---:|
| scene-0770 | 105 | -0.363 | -3.540 | 0.969 |
| scene-0398 | 94 | -2.078 | -2.605 | 0.867 |
| scene-0184 | 45 | -3.042 | -2.997 | 0.922 |
| scene-0245 | 103 | -1.419 | -1.769 | 0.750 |

## Accounting for the observed tail-gap change

Entries are MPS-on minus MPS-off, in milliseconds. For each run, each component is evaluated on the same interpolated frames defining the total P50 and P99. The three component columns add to the total change, but this is descriptive endpoint accounting, not an intervention or a module's own percentile gap.

| Scene | Total gap change | Voxelization contribution change | Other kernels contribution change | Nonkernel contribution change |
|---|---:|---:|---:|---:|
| scene-0770 | -0.363 | 0.312 | -0.131 | -0.544 |
| scene-0398 | -2.078 | -3.480 | 0.151 | 1.251 |
| scene-0184 | -3.042 | -6.150 | 1.254 | 1.854 |
| scene-0245 | -1.419 | 0.226 | -0.320 | -1.325 |

Each condition has one execution and temporally dependent observations. All full-sample P99 estimates have fewer than 1,000 observations; common-frame subsets below 100 are especially fragile. Mode differences remain observational because execution order, runtime state and frame timing are not independently replicated. A module's contribution is part of total latency, so high correlation alone is not causal proof.

The input-driven voxelization mechanism is supported by implementation inspection, reconstructed workload and recorded kernel timings. The precise cause of occasional stalls and the smaller MPS-versus-disabled tail difference remains unresolved. No finer neural-network layer labels are inferred from the top-level annotations. Resolving the residual mechanism would require targeted repeated timing and launch/memory/scheduler evidence; no additional GPU or layer-detail runs were scheduled.

`associations.csv` also reports additive changes at the total latency's P50/P99 endpoint frames. They sum to total P99-P50, but are not each module's independent P99-P50 and can be negative. `nonkernel_ms` includes memory copies, host work, launch gaps and synchronization, not exclusively CPU execution.

Reproduce after sourcing ROS and the workspace:

```bash
python3 -m closeloop_analyzer.input_data.centerpoint_diagnosis /mmdetection3d_ros2/analysis_outputs/input2
python3 -m closeloop_profiler.centerpoint_input_geometry /mmdetection3d_ros2/analysis_outputs/input2
python3 -m closeloop_analyzer.input_data.centerpoint_diagnosis /mmdetection3d_ros2/analysis_outputs/input2 --report-only
```
