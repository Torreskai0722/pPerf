# input1-two-pass-physical-rain descriptive analysis

## Scope and completion

Scope: **single-run exploratory descriptive**. One execution was used per condition. Frames within an execution are repeated observations, not independent experimental repetitions. No accuracy or confidence claim is made.

Pair campaign: **210/210 valid**; missing or invalid: **0**. Isolated controls were outside this campaign and are not analyzed.

The pair matrix crosses 3 LiDAR models with 7 camera models, 5 same-condition inputs, and MPS modes off, on: 210 cells. Both tenants consume the same clear or adverse bag; labels below compare each adverse cell only with its configured clear baseline.

## Actual input changes

All values below come from decoded payloads in the immutable MCAP bags. Physical-rain conditions are deterministic corruptions of the listed full-duration clear source and retain source timestamps.

### Clear-condition input context

| condition | image luminance mean | image horizontal gradient | LiDAR points | LiDAR mean range (m) |
|---|---:|---:|---:|---:|
| clear | 100.835 | 1.655 | 34720.000 | 8.409 |

### Paired adverse-minus-clear payload changes

`median paired change` is computed at identical source timestamps. Ratios use adverse median divided by matching clear median.

| condition | modality | metric | paired n | clear median | adverse median | median paired change | ratio |
|---|---|---|---:|---:|---:|---:|---:|
| r7p5 | image | horizontal_gradient_mean | 229 | 1.655 | 1.730 | 0.069 | 1.045 |
| r7p5 | image | jpeg_bytes | 229 | 121789.000 | 218677.000 | 96969.000 | 1.796 |
| r7p5 | image | luminance_mean | 229 | 100.835 | 100.325 | -0.514 | 0.995 |
| r7p5 | image | luminance_std | 229 | 61.710 | 60.220 | -1.361 | 0.976 |
| r7p5 | image | vertical_gradient_mean | 229 | 2.291 | 2.254 | -0.043 | 0.984 |
| r7p5 | lidar | intensity_mean | 384 | 15.271 | 15.433 | 0.159 | 1.011 |
| r7p5 | lidar | point_count | 384 | 34720.000 | 33186.500 | -1531.500 | 0.956 |
| r7p5 | lidar | points_0_5m | 384 | 15549.000 | 15148.000 | -204.000 | 0.974 |
| r7p5 | lidar | points_20_40m | 384 | 2100.000 | 1668.000 | -448.000 | 0.794 |
| r7p5 | lidar | points_40m_plus | 384 | 682.000 | 619.000 | -61.000 | 0.908 |
| r7p5 | lidar | points_5_20m | 384 | 16422.000 | 15721.500 | -731.500 | 0.957 |
| r7p5 | lidar | range_mean_m | 384 | 8.409 | 8.097 | -0.372 | 0.963 |
| r15 | image | horizontal_gradient_mean | 229 | 1.655 | 1.837 | 0.178 | 1.110 |
| r15 | image | jpeg_bytes | 229 | 121789.000 | 226097.000 | 104416.000 | 1.856 |
| r15 | image | luminance_mean | 229 | 100.835 | 100.340 | -0.492 | 0.995 |
| r15 | image | luminance_std | 229 | 61.710 | 59.379 | -2.124 | 0.962 |
| r15 | image | vertical_gradient_mean | 229 | 2.291 | 2.237 | -0.063 | 0.977 |
| r15 | lidar | intensity_mean | 384 | 15.271 | 15.179 | -0.092 | 0.994 |
| r15 | lidar | point_count | 384 | 34720.000 | 33159.000 | -1559.500 | 0.955 |
| r15 | lidar | points_0_5m | 384 | 15549.000 | 15237.500 | -111.000 | 0.980 |
| r15 | lidar | points_20_40m | 384 | 2100.000 | 1630.000 | -496.000 | 0.776 |
| r15 | lidar | points_40m_plus | 384 | 682.000 | 599.000 | -81.000 | 0.878 |
| r15 | lidar | points_5_20m | 384 | 16422.000 | 15669.000 | -781.500 | 0.954 |
| r15 | lidar | range_mean_m | 384 | 8.409 | 8.020 | -0.450 | 0.954 |
| r25 | image | horizontal_gradient_mean | 229 | 1.655 | 1.986 | 0.323 | 1.200 |
| r25 | image | jpeg_bytes | 229 | 121789.000 | 235503.000 | 113556.000 | 1.934 |
| r25 | image | luminance_mean | 229 | 100.835 | 100.376 | -0.464 | 0.995 |
| r25 | image | luminance_std | 229 | 61.710 | 58.480 | -2.941 | 0.948 |
| r25 | image | vertical_gradient_mean | 229 | 2.291 | 2.220 | -0.082 | 0.969 |
| r25 | lidar | intensity_mean | 384 | 15.271 | 14.933 | -0.337 | 0.978 |
| r25 | lidar | point_count | 384 | 34720.000 | 33093.000 | -1633.500 | 0.953 |
| r25 | lidar | points_0_5m | 384 | 15549.000 | 15325.500 | -21.500 | 0.986 |
| r25 | lidar | points_20_40m | 384 | 2100.000 | 1597.500 | -534.000 | 0.761 |
| r25 | lidar | points_40m_plus | 384 | 682.000 | 546.500 | -137.000 | 0.801 |
| r25 | lidar | points_5_20m | 384 | 16422.000 | 15612.500 | -835.000 | 0.951 |
| r25 | lidar | range_mean_m | 384 | 8.409 | 7.920 | -0.574 | 0.942 |
| r50 | image | horizontal_gradient_mean | 229 | 1.655 | 2.360 | 0.692 | 1.426 |
| r50 | image | jpeg_bytes | 229 | 121789.000 | 255146.000 | 133419.000 | 2.095 |
| r50 | image | luminance_mean | 229 | 100.835 | 100.388 | -0.453 | 0.996 |
| r50 | image | luminance_std | 229 | 61.710 | 56.847 | -4.411 | 0.921 |
| r50 | image | vertical_gradient_mean | 229 | 2.291 | 2.196 | -0.106 | 0.959 |
| r50 | lidar | intensity_mean | 384 | 15.271 | 14.456 | -0.826 | 0.947 |
| r50 | lidar | point_count | 384 | 34720.000 | 32837.500 | -1888.000 | 0.946 |
| r50 | lidar | points_0_5m | 384 | 15549.000 | 15520.500 | 160.000 | 0.998 |
| r50 | lidar | points_20_40m | 384 | 2100.000 | 1537.000 | -588.500 | 0.732 |
| r50 | lidar | points_40m_plus | 384 | 682.000 | 292.000 | -389.000 | 0.428 |
| r50 | lidar | points_5_20m | 384 | 16422.000 | 15490.000 | -953.000 | 0.943 |
| r50 | lidar | range_mean_m | 384 | 8.409 | 7.414 | -1.039 | 0.882 |

## Isolated-model computational response

Not produced: isolated controls were outside the v4 campaign.

| model | adverse condition | median latency change (ms) | clear minimum-P99 range (ms) | adverse minimum-P99 range (ms) | factor | label |
|---|---|---:|---:|---:|---:|---|

The latency table above stores width comparisons; exact median changes and compute-path measurements are in `isolated_mechanism_comparisons.csv`.

### Isolated compute-path summary across adverse conditions

| model | metric | observations | median adverse-clear median change | amplified | damped | unchanged |
|---|---|---:|---:|---:|---:|---:|

### Module execution summary

| model | module | adverse comparisons | median duration change (ms) | amplified widths | damped widths | unchanged widths |
|---|---|---:|---:|---:|---:|---:|

## Controlled synthetic-input validation

This table holds scene, model, weights, MPS-off isolation, and source timestamp alignment fixed while the deterministic weather payload changes. It validates whether computation changed alongside the measured input; it does not establish that a single measured input feature caused the computation change.

| model | condition | input metric | input ratio | latency median change (ms) | latency width factor | GPU-active median change (ms) | kernel-count median change | dynamic metric change |
|---|---|---|---:|---:|---:|---:|---:|---|

## Same-condition pair amplification and damping

Every label in this section compares an adverse pair only with its matching clear pair under the same MPS mode. Isolated widths are not used as denominators.

| target | MPS | adverse pair observations | amplified | damped | unchanged | median width factor | minimum factor | maximum factor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3dssd | off | 28 | 5 | 23 | 0 | 0.982 | 0.862 | 1.066 |
| 3dssd | on | 28 | 20 | 8 | 0 | 1.024 | 0.942 | 1.210 |
| centerpoint | off | 28 | 3 | 25 | 0 | 0.925 | 0.611 | 1.022 |
| centerpoint | on | 28 | 12 | 16 | 0 | 0.988 | 0.829 | 1.093 |
| deeplabv3plus | off | 12 | 4 | 8 | 0 | 0.988 | 0.964 | 1.016 |
| deeplabv3plus | on | 12 | 6 | 6 | 0 | 0.991 | 0.921 | 1.057 |
| detr | off | 12 | 10 | 2 | 0 | 1.023 | 0.992 | 1.054 |
| detr | on | 12 | 6 | 6 | 0 | 0.991 | 0.895 | 1.057 |
| dino | off | 12 | 12 | 0 | 0 | 1.022 | 1.001 | 1.301 |
| dino | on | 12 | 4 | 8 | 0 | 0.954 | 0.910 | 1.474 |
| faster-rcnn | off | 12 | 11 | 1 | 0 | 1.028 | 0.992 | 1.059 |
| faster-rcnn | on | 12 | 7 | 5 | 0 | 1.016 | 0.913 | 1.079 |
| mask-rcnn | off | 12 | 5 | 7 | 0 | 0.988 | 0.929 | 1.065 |
| mask-rcnn | on | 12 | 3 | 9 | 0 | 0.963 | 0.895 | 1.066 |
| pointpillars | off | 28 | 14 | 14 | 0 | 1.002 | 0.427 | 1.044 |
| pointpillars | on | 28 | 12 | 16 | 0 | 0.995 | 0.899 | 1.216 |
| vit-upernet | off | 12 | 10 | 2 | 0 | 1.016 | 0.983 | 1.049 |
| vit-upernet | on | 12 | 2 | 10 | 0 | 0.967 | 0.919 | 1.042 |
| yolov3 | off | 12 | 5 | 7 | 0 | 0.994 | 0.805 | 1.040 |
| yolov3 | on | 12 | 4 | 8 | 0 | 0.951 | 0.908 | 1.114 |

### Co-runner-specific latency-width response

| pair | target | MPS | conditions | amplified | damped | unchanged | median factor |
|---|---|---|---:|---:|---:|---:|---:|
| 3dssd+deeplabv3plus | 3dssd | off | 4 | 0 | 4 | 0 | 0.950 |
| 3dssd+deeplabv3plus | 3dssd | on | 4 | 4 | 0 | 0 | 1.032 |
| 3dssd+deeplabv3plus | deeplabv3plus | off | 4 | 1 | 3 | 0 | 0.984 |
| 3dssd+deeplabv3plus | deeplabv3plus | on | 4 | 3 | 1 | 0 | 1.012 |
| 3dssd+detr | 3dssd | off | 4 | 0 | 4 | 0 | 0.964 |
| 3dssd+detr | 3dssd | on | 4 | 4 | 0 | 0 | 1.080 |
| 3dssd+detr | detr | off | 4 | 3 | 1 | 0 | 1.025 |
| 3dssd+detr | detr | on | 4 | 4 | 0 | 0 | 1.035 |
| 3dssd+dino | 3dssd | off | 4 | 2 | 2 | 0 | 0.995 |
| 3dssd+dino | 3dssd | on | 4 | 4 | 0 | 0 | 1.041 |
| 3dssd+dino | dino | off | 4 | 4 | 0 | 0 | 1.022 |
| 3dssd+dino | dino | on | 4 | 4 | 0 | 0 | 1.368 |
| 3dssd+faster-rcnn | 3dssd | off | 4 | 0 | 4 | 0 | 0.988 |
| 3dssd+faster-rcnn | 3dssd | on | 4 | 3 | 1 | 0 | 1.001 |
| 3dssd+faster-rcnn | faster-rcnn | off | 4 | 3 | 1 | 0 | 1.013 |
| 3dssd+faster-rcnn | faster-rcnn | on | 4 | 4 | 0 | 0 | 1.056 |
| 3dssd+mask-rcnn | 3dssd | off | 4 | 2 | 2 | 0 | 0.953 |
| 3dssd+mask-rcnn | 3dssd | on | 4 | 0 | 4 | 0 | 0.987 |
| 3dssd+mask-rcnn | mask-rcnn | off | 4 | 3 | 1 | 0 | 1.020 |
| 3dssd+mask-rcnn | mask-rcnn | on | 4 | 1 | 3 | 0 | 0.974 |
| 3dssd+vit-upernet | 3dssd | off | 4 | 0 | 4 | 0 | 0.977 |
| 3dssd+vit-upernet | 3dssd | on | 4 | 1 | 3 | 0 | 0.991 |
| 3dssd+vit-upernet | vit-upernet | off | 4 | 4 | 0 | 0 | 1.016 |
| 3dssd+vit-upernet | vit-upernet | on | 4 | 2 | 2 | 0 | 0.994 |
| 3dssd+yolov3 | 3dssd | off | 4 | 1 | 3 | 0 | 0.972 |
| 3dssd+yolov3 | 3dssd | on | 4 | 4 | 0 | 0 | 1.117 |
| 3dssd+yolov3 | yolov3 | off | 4 | 3 | 1 | 0 | 1.009 |
| 3dssd+yolov3 | yolov3 | on | 4 | 3 | 1 | 0 | 1.076 |
| centerpoint+deeplabv3plus | centerpoint | off | 4 | 0 | 4 | 0 | 0.955 |
| centerpoint+deeplabv3plus | centerpoint | on | 4 | 1 | 3 | 0 | 0.980 |
| centerpoint+deeplabv3plus | deeplabv3plus | off | 4 | 1 | 3 | 0 | 0.990 |
| centerpoint+deeplabv3plus | deeplabv3plus | on | 4 | 0 | 4 | 0 | 0.923 |
| centerpoint+detr | centerpoint | off | 4 | 0 | 4 | 0 | 0.863 |
| centerpoint+detr | centerpoint | on | 4 | 0 | 4 | 0 | 0.938 |
| centerpoint+detr | detr | off | 4 | 3 | 1 | 0 | 1.032 |
| centerpoint+detr | detr | on | 4 | 0 | 4 | 0 | 0.910 |
| centerpoint+dino | centerpoint | off | 4 | 0 | 4 | 0 | 0.798 |
| centerpoint+dino | centerpoint | on | 4 | 0 | 4 | 0 | 0.978 |
| centerpoint+dino | dino | off | 4 | 4 | 0 | 0 | 1.017 |
| centerpoint+dino | dino | on | 4 | 0 | 4 | 0 | 0.946 |
| centerpoint+faster-rcnn | centerpoint | off | 4 | 0 | 4 | 0 | 0.909 |
| centerpoint+faster-rcnn | centerpoint | on | 4 | 4 | 0 | 0 | 1.022 |
| centerpoint+faster-rcnn | faster-rcnn | off | 4 | 4 | 0 | 0 | 1.021 |
| centerpoint+faster-rcnn | faster-rcnn | on | 4 | 0 | 4 | 0 | 0.928 |
| centerpoint+mask-rcnn | centerpoint | off | 4 | 0 | 4 | 0 | 0.892 |
| centerpoint+mask-rcnn | centerpoint | on | 4 | 3 | 1 | 0 | 1.008 |
| centerpoint+mask-rcnn | mask-rcnn | off | 4 | 0 | 4 | 0 | 0.947 |
| centerpoint+mask-rcnn | mask-rcnn | on | 4 | 0 | 4 | 0 | 0.954 |
| centerpoint+vit-upernet | centerpoint | off | 4 | 3 | 1 | 0 | 1.004 |
| centerpoint+vit-upernet | centerpoint | on | 4 | 4 | 0 | 0 | 1.043 |
| centerpoint+vit-upernet | vit-upernet | off | 4 | 3 | 1 | 0 | 1.010 |
| centerpoint+vit-upernet | vit-upernet | on | 4 | 0 | 4 | 0 | 0.945 |
| centerpoint+yolov3 | centerpoint | off | 4 | 0 | 4 | 0 | 0.900 |
| centerpoint+yolov3 | centerpoint | on | 4 | 0 | 4 | 0 | 0.929 |
| centerpoint+yolov3 | yolov3 | off | 4 | 2 | 2 | 0 | 0.997 |
| centerpoint+yolov3 | yolov3 | on | 4 | 0 | 4 | 0 | 0.931 |
| pointpillars+deeplabv3plus | deeplabv3plus | off | 4 | 2 | 2 | 0 | 0.996 |
| pointpillars+deeplabv3plus | deeplabv3plus | on | 4 | 3 | 1 | 0 | 1.013 |
| pointpillars+deeplabv3plus | pointpillars | off | 4 | 2 | 2 | 0 | 1.005 |
| pointpillars+deeplabv3plus | pointpillars | on | 4 | 0 | 4 | 0 | 0.942 |
| pointpillars+detr | detr | off | 4 | 4 | 0 | 0 | 1.022 |
| pointpillars+detr | detr | on | 4 | 2 | 2 | 0 | 0.991 |
| pointpillars+detr | pointpillars | off | 4 | 4 | 0 | 0 | 1.013 |
| pointpillars+detr | pointpillars | on | 4 | 2 | 2 | 0 | 0.985 |
| pointpillars+dino | dino | off | 4 | 4 | 0 | 0 | 1.252 |
| pointpillars+dino | dino | on | 4 | 0 | 4 | 0 | 0.932 |
| pointpillars+dino | pointpillars | off | 4 | 0 | 4 | 0 | 0.979 |
| pointpillars+dino | pointpillars | on | 4 | 0 | 4 | 0 | 0.983 |
| pointpillars+faster-rcnn | faster-rcnn | off | 4 | 4 | 0 | 0 | 1.046 |
| pointpillars+faster-rcnn | faster-rcnn | on | 4 | 3 | 1 | 0 | 1.016 |
| pointpillars+faster-rcnn | pointpillars | off | 4 | 0 | 4 | 0 | 0.958 |
| pointpillars+faster-rcnn | pointpillars | on | 4 | 2 | 2 | 0 | 0.999 |
| pointpillars+mask-rcnn | mask-rcnn | off | 4 | 2 | 2 | 0 | 1.012 |
| pointpillars+mask-rcnn | mask-rcnn | on | 4 | 2 | 2 | 0 | 1.013 |
| pointpillars+mask-rcnn | pointpillars | off | 4 | 1 | 3 | 0 | 0.988 |
| pointpillars+mask-rcnn | pointpillars | on | 4 | 1 | 3 | 0 | 0.990 |
| pointpillars+vit-upernet | pointpillars | off | 4 | 4 | 0 | 0 | 1.010 |
| pointpillars+vit-upernet | pointpillars | on | 4 | 3 | 1 | 0 | 1.009 |
| pointpillars+vit-upernet | vit-upernet | off | 4 | 3 | 1 | 0 | 1.022 |
| pointpillars+vit-upernet | vit-upernet | on | 4 | 0 | 4 | 0 | 0.986 |
| pointpillars+yolov3 | pointpillars | off | 4 | 3 | 1 | 0 | 1.026 |
| pointpillars+yolov3 | pointpillars | on | 4 | 4 | 0 | 0 | 1.149 |
| pointpillars+yolov3 | yolov3 | off | 4 | 0 | 4 | 0 | 0.845 |
| pointpillars+yolov3 | yolov3 | on | 4 | 1 | 3 | 0 | 0.944 |

## Compute, kernel, waiting, co-runner, and MPS mechanisms

`gpu_kernel_active_ms` measures summed kernel service, `kernel_span_ms` measures first-to-last kernel extent, `waiting_ms` is the captured memcpy-adjacent non-kernel interval, and `kernel_count` records launch structure. Their joint movement is descriptive evidence: active-time or count changes are consistent with input-driven work changes; span and waiting changes without matching active-time changes are consistent with scheduling/interference. These measurements do not identify a specific GPU resource.

| target | MPS | metric | observations | median adverse-clear median change | amplified widths | damped widths | unchanged widths |
|---|---|---|---:|---:|---:|---:|---:|
| 3dssd | off | gpu_kernel_active_ms | 28 | 0.015 | 8 | 20 | 0 |
| 3dssd | off | inference_e2e_ms | 28 | 0.255 | 5 | 23 | 0 |
| 3dssd | off | input_point_count | 28 | -1594.500 | 28 | 0 | 0 |
| 3dssd | off | kernel_count | 28 | 0.000 | 0 | 0 | 28 |
| 3dssd | off | kernel_span_ms | 28 | 0.305 | 4 | 24 | 0 |
| 3dssd | off | sampled_point_count | 28 | 0.000 | 0 | 0 | 0 |
| 3dssd | off | waiting_ms | 28 | 0.229 | 4 | 24 | 0 |
| 3dssd | on | gpu_kernel_active_ms | 28 | -0.065 | 14 | 14 | 0 |
| 3dssd | on | inference_e2e_ms | 28 | -0.181 | 20 | 8 | 0 |
| 3dssd | on | input_point_count | 28 | -1594.000 | 28 | 0 | 0 |
| 3dssd | on | kernel_count | 28 | 0.000 | 0 | 0 | 28 |
| 3dssd | on | kernel_span_ms | 28 | -0.146 | 18 | 10 | 0 |
| 3dssd | on | sampled_point_count | 28 | 0.000 | 0 | 0 | 0 |
| 3dssd | on | waiting_ms | 28 | -0.075 | 20 | 8 | 0 |
| centerpoint | off | gpu_kernel_active_ms | 28 | -6.422 | 5 | 23 | 0 |
| centerpoint | off | inference_e2e_ms | 28 | -9.544 | 3 | 25 | 0 |
| centerpoint | off | input_point_count | 28 | -1597.000 | 28 | 0 | 0 |
| centerpoint | off | kernel_count | 28 | -12.000 | 2 | 14 | 12 |
| centerpoint | off | kernel_span_ms | 28 | -9.422 | 3 | 25 | 0 |
| centerpoint | off | occupied_voxel_count | 28 | -509.750 | 28 | 0 | 0 |
| centerpoint | off | waiting_ms | 28 | -0.661 | 8 | 20 | 0 |
| centerpoint | on | gpu_kernel_active_ms | 28 | -8.292 | 15 | 13 | 0 |
| centerpoint | on | inference_e2e_ms | 28 | -7.882 | 12 | 16 | 0 |
| centerpoint | on | input_point_count | 28 | -1593.500 | 28 | 0 | 0 |
| centerpoint | on | kernel_count | 28 | -12.000 | 1 | 12 | 15 |
| centerpoint | on | kernel_span_ms | 28 | -7.706 | 12 | 16 | 0 |
| centerpoint | on | occupied_voxel_count | 28 | -584.500 | 28 | 0 | 0 |
| centerpoint | on | waiting_ms | 28 | -0.376 | 15 | 13 | 0 |
| deeplabv3plus | off | gpu_kernel_active_ms | 12 | 0.031 | 5 | 7 | 0 |
| deeplabv3plus | off | inference_e2e_ms | 12 | 0.378 | 4 | 8 | 0 |
| deeplabv3plus | off | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| deeplabv3plus | off | kernel_span_ms | 12 | 0.238 | 5 | 7 | 0 |
| deeplabv3plus | off | waiting_ms | 12 | -0.061 | 7 | 5 | 0 |
| deeplabv3plus | on | gpu_kernel_active_ms | 12 | 0.222 | 6 | 6 | 0 |
| deeplabv3plus | on | inference_e2e_ms | 12 | 0.468 | 6 | 6 | 0 |
| deeplabv3plus | on | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| deeplabv3plus | on | kernel_span_ms | 12 | 0.381 | 6 | 6 | 0 |
| deeplabv3plus | on | waiting_ms | 12 | -0.001 | 1 | 11 | 0 |
| detr | off | gpu_kernel_active_ms | 12 | 0.016 | 10 | 2 | 0 |
| detr | off | inference_e2e_ms | 12 | -1.015 | 10 | 2 | 0 |
| detr | off | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| detr | off | kernel_span_ms | 12 | -0.255 | 7 | 5 | 0 |
| detr | off | waiting_ms | 12 | -1.507 | 4 | 8 | 0 |
| detr | on | gpu_kernel_active_ms | 12 | -0.320 | 7 | 5 | 0 |
| detr | on | inference_e2e_ms | 12 | -0.491 | 6 | 6 | 0 |
| detr | on | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| detr | on | kernel_span_ms | 12 | -0.357 | 5 | 7 | 0 |
| detr | on | waiting_ms | 12 | -0.539 | 6 | 6 | 0 |
| dino | off | gpu_kernel_active_ms | 12 | 0.085 | 5 | 7 | 0 |
| dino | off | inference_e2e_ms | 12 | 1.001 | 12 | 0 | 0 |
| dino | off | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| dino | off | kernel_span_ms | 12 | 1.034 | 12 | 0 | 0 |
| dino | off | waiting_ms | 12 | 0.569 | 4 | 8 | 0 |
| dino | on | gpu_kernel_active_ms | 12 | 0.425 | 10 | 2 | 0 |
| dino | on | inference_e2e_ms | 12 | 0.513 | 4 | 8 | 0 |
| dino | on | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| dino | on | kernel_span_ms | 12 | 0.803 | 5 | 7 | 0 |
| dino | on | waiting_ms | 12 | 0.239 | 5 | 7 | 0 |
| faster-rcnn | off | gpu_kernel_active_ms | 12 | -0.010 | 9 | 3 | 0 |
| faster-rcnn | off | inference_e2e_ms | 12 | -2.121 | 11 | 1 | 0 |
| faster-rcnn | off | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| faster-rcnn | off | kernel_span_ms | 12 | -2.328 | 11 | 1 | 0 |
| faster-rcnn | off | proposal_count | 12 | 0.000 | 0 | 0 | 0 |
| faster-rcnn | off | waiting_ms | 12 | -2.304 | 8 | 4 | 0 |
| faster-rcnn | on | gpu_kernel_active_ms | 12 | -0.101 | 10 | 2 | 0 |
| faster-rcnn | on | inference_e2e_ms | 12 | -2.485 | 7 | 5 | 0 |
| faster-rcnn | on | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| faster-rcnn | on | kernel_span_ms | 12 | -0.310 | 6 | 6 | 0 |
| faster-rcnn | on | proposal_count | 12 | 0.000 | 0 | 0 | 0 |
| faster-rcnn | on | waiting_ms | 12 | -2.408 | 2 | 10 | 0 |
| mask-rcnn | off | gpu_kernel_active_ms | 12 | -0.112 | 4 | 8 | 0 |
| mask-rcnn | off | inference_e2e_ms | 12 | -0.134 | 5 | 7 | 0 |
| mask-rcnn | off | kernel_count | 12 | 0.000 | 0 | 0 | 12 |
| mask-rcnn | off | kernel_span_ms | 12 | -0.171 | 5 | 7 | 0 |
| mask-rcnn | off | proposal_count | 12 | 0.000 | 0 | 0 | 0 |
| mask-rcnn | off | waiting_ms | 12 | 0.543 | 9 | 3 | 0 |
| mask-rcnn | on | gpu_kernel_active_ms | 12 | -0.553 | 1 | 11 | 0 |
| mask-rcnn | on | inference_e2e_ms | 12 | -1.055 | 3 | 9 | 0 |
| mask-rcnn | on | kernel_count | 12 | 0.000 | 0 | 0 | 12 |
| mask-rcnn | on | kernel_span_ms | 12 | -1.287 | 3 | 9 | 0 |
| mask-rcnn | on | proposal_count | 12 | 0.000 | 0 | 0 | 0 |
| mask-rcnn | on | waiting_ms | 12 | -0.921 | 3 | 9 | 0 |
| pointpillars | off | gpu_kernel_active_ms | 28 | -0.379 | 6 | 22 | 0 |
| pointpillars | off | inference_e2e_ms | 28 | -0.129 | 14 | 14 | 0 |
| pointpillars | off | input_point_count | 28 | -1595.000 | 28 | 0 | 0 |
| pointpillars | off | kernel_count | 28 | 0.000 | 0 | 0 | 28 |
| pointpillars | off | kernel_span_ms | 28 | 0.042 | 16 | 12 | 0 |
| pointpillars | off | occupied_voxel_count | 28 | -40.500 | 0 | 28 | 0 |
| pointpillars | off | waiting_ms | 28 | 0.812 | 9 | 19 | 0 |
| pointpillars | on | gpu_kernel_active_ms | 28 | -0.804 | 5 | 23 | 0 |
| pointpillars | on | inference_e2e_ms | 28 | -1.180 | 12 | 16 | 0 |
| pointpillars | on | input_point_count | 28 | -1594.000 | 28 | 0 | 0 |
| pointpillars | on | kernel_count | 28 | 0.000 | 0 | 0 | 28 |
| pointpillars | on | kernel_span_ms | 28 | -1.301 | 8 | 20 | 0 |
| pointpillars | on | occupied_voxel_count | 28 | -56.500 | 0 | 28 | 0 |
| pointpillars | on | waiting_ms | 28 | -1.163 | 12 | 16 | 0 |
| vit-upernet | off | gpu_kernel_active_ms | 12 | 0.002 | 9 | 3 | 0 |
| vit-upernet | off | inference_e2e_ms | 12 | 0.353 | 10 | 2 | 0 |
| vit-upernet | off | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| vit-upernet | off | kernel_span_ms | 12 | 0.167 | 10 | 2 | 0 |
| vit-upernet | off | waiting_ms | 12 | -0.124 | 6 | 6 | 0 |
| vit-upernet | on | gpu_kernel_active_ms | 12 | -0.063 | 6 | 6 | 0 |
| vit-upernet | on | inference_e2e_ms | 12 | -2.030 | 2 | 10 | 0 |
| vit-upernet | on | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| vit-upernet | on | kernel_span_ms | 12 | -0.183 | 2 | 10 | 0 |
| vit-upernet | on | waiting_ms | 12 | 0.000 | 7 | 5 | 0 |
| yolov3 | off | gpu_kernel_active_ms | 12 | -0.004 | 5 | 7 | 0 |
| yolov3 | off | inference_e2e_ms | 12 | -0.581 | 5 | 7 | 0 |
| yolov3 | off | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| yolov3 | off | kernel_span_ms | 12 | 0.372 | 5 | 7 | 0 |
| yolov3 | off | waiting_ms | 12 | -0.383 | 4 | 8 | 0 |
| yolov3 | on | gpu_kernel_active_ms | 12 | 0.013 | 7 | 5 | 0 |
| yolov3 | on | inference_e2e_ms | 12 | 0.329 | 4 | 8 | 0 |
| yolov3 | on | kernel_count | 12 | 0.000 | 0 | 0 | 0 |
| yolov3 | on | kernel_span_ms | 12 | 0.377 | 4 | 8 | 0 |
| yolov3 | on | waiting_ms | 12 | 0.368 | 8 | 4 | 0 |

### Intrinsic-versus-pair label agreement

| pair | target | MPS | latency comparisons | same isolated/pair label | different label |
|---|---|---|---:|---:|---:|

### MPS label transitions

MPS rows place the already valid within-mode off/on comparisons side by side; they do not compare raw adverse latency across modes.

| target | off label → on label | observations |
|---|---|---:|
| 3dssd | observed_amplification → observed_amplification | 3 |
| 3dssd | observed_amplification → observed_damping | 2 |
| 3dssd | observed_damping → observed_amplification | 17 |
| 3dssd | observed_damping → observed_damping | 6 |
| centerpoint | observed_amplification → observed_amplification | 3 |
| centerpoint | observed_damping → observed_amplification | 9 |
| centerpoint | observed_damping → observed_damping | 16 |
| deeplabv3plus | observed_amplification → observed_amplification | 1 |
| deeplabv3plus | observed_amplification → observed_damping | 3 |
| deeplabv3plus | observed_damping → observed_amplification | 5 |
| deeplabv3plus | observed_damping → observed_damping | 3 |
| detr | observed_amplification → observed_amplification | 5 |
| detr | observed_amplification → observed_damping | 5 |
| detr | observed_damping → observed_amplification | 1 |
| detr | observed_damping → observed_damping | 1 |
| dino | observed_amplification → observed_amplification | 4 |
| dino | observed_amplification → observed_damping | 8 |
| faster-rcnn | observed_amplification → observed_amplification | 6 |
| faster-rcnn | observed_amplification → observed_damping | 5 |
| faster-rcnn | observed_damping → observed_amplification | 1 |
| mask-rcnn | observed_amplification → observed_amplification | 3 |
| mask-rcnn | observed_amplification → observed_damping | 2 |
| mask-rcnn | observed_damping → observed_damping | 7 |
| pointpillars | observed_amplification → observed_amplification | 8 |
| pointpillars | observed_amplification → observed_damping | 6 |
| pointpillars | observed_damping → observed_amplification | 4 |
| pointpillars | observed_damping → observed_damping | 10 |
| vit-upernet | observed_amplification → observed_amplification | 2 |
| vit-upernet | observed_amplification → observed_damping | 8 |
| vit-upernet | observed_damping → observed_damping | 2 |
| yolov3 | observed_amplification → observed_amplification | 3 |
| yolov3 | observed_amplification → observed_damping | 2 |
| yolov3 | observed_damping → observed_amplification | 1 |
| yolov3 | observed_damping → observed_damping | 6 |

## Architecture-family interpretation boundary

- 3DSSD is point-based: raw point count/range changes can reach preprocessing and early point operations; `sampled_point_count` shows whether its captured final sampling cardinality changed.
- CenterPoint is voxel-based: `occupied_voxel_count` separates raw point loss/addition from the realized sparse voxel workload.
- PointPillars is voxel/pillar-based: `occupied_voxel_count` and its voxel encoder, middle encoder, backbone, neck, and head durations describe the realized pillar workload.
- DeepLabV3+ and ViT-UPerNet are dense segmentation paths with fixed configured resize shapes; input sensitivity is assessed through preprocessing, module durations, kernel activity/count/span, and waiting rather than a proposal count.
- DETR uses a transformer detector path with fixed configured input processing and query structure; the same fixed/dynamic measurements bound any observed response.
- DINO uses the same measured transformer stages: backbone, neck, positional encoding, encoder, decoder, and bbox head. Warm-up must observe every required binding before a run is accepted.
- Faster R-CNN is proposal-based: `proposal_count`, RPN, and ROI module durations expose its captured dynamic post-backbone path.
- YOLOv3 is single-stage: no proposal-stage count is expected, so preprocessing and dense backbone/head GPU measurements are used.

These are architecture-aware interpretations of measured execution structure, not accuracy claims or proof that weather alone caused any co-run latency change.

## Completeness, exclusions, failures, and limitations

No planned run was excluded or invalid.

Limitations:

- One execution per cell supports descriptive comparisons only; within-run frames do not provide run-to-run uncertainty.
- Rain and snow are deterministic synthetic corruptions, not a claim about every natural weather process.
- The study measures computation and latency, not prediction accuracy.
- Pair changes combine both tenants' input responses and co-run effects; without isolated controls they cannot be separated.
- Nsight instrumentation perturbs timing; all matrix cells use the same profiling path, and conclusions remain descriptive.
- No specific shared GPU resource is named without direct evidence.

## Artifacts and regeneration

- Study contract: `/mmdetection3d_ros2/closeloop_perf/studies/input1/study.yaml` (SHA-256 `945f8b29ad82e26ac8d398649d443cf214bb4626ac9a1d4adab9385093916ae4`).
- Pair and new 3DSSD raw runs: `/mmdetection3d_ros2/analysis_outputs/input1/runs/<run_id>`.
- Preserved failed attempts: `/mmdetection3d_ros2/analysis_outputs/input1/runs/failed_attempts/<run-id>-attempt-N`.
- Isolated raw runs are listed in `isolated_validation.csv` when included.
- Analysis tables, plots, manifest, and this report: `/mmdetection3d_ros2/analysis_outputs/input1/analysis/input1-two-pass-physical-rain/input-data`.
- Each Goal 1 raw directory retains `config.yaml`, run/model/status evidence, input JSONL, clock evidence, `profile.nsys-rep.gz`, `profile.sqlite.gz`, and `profile_archive.json`. Compression is checksum-verified and lossless.

Run from `/mmdetection3d_ros2` inside `pPerf-host`:

```bash
source /opt/ros/humble/setup.bash
source closeloop_perf/install/setup.bash
python3 -m closeloop_analyzer.input_data.corrected_input_analysis /mmdetection3d_ros2/closeloop_perf/studies/input1/study.yaml --output-root /mmdetection3d_ros2/analysis_outputs/input1/analysis/input1-two-pass-physical-rain/input-data --pair-only
```

Restore one archived trace without replacing the archive:

```bash
gzip -dk RUN/profile.sqlite.gz
gzip -dk RUN/profile.nsys-rep.gz
```

All per-frame, per-module, per-structure, controlled-response, pair context, ECDF, and comparison rows are retained in the CSV artifacts; the tables above summarize rather than silently exclude them.
