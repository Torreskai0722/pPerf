# Controlled environment sensitivity

Validated analysis rows: 376. Blocked execution slots: 0.

[Single-model baseline violin plots by scene](isolated_baselines.md) ([all models as PDF](plots/isolated/isolated_baselines.pdf)).

All plots and performance statistics use one inclusive P1–P99 latency filter per execution/model, with original cutoffs computed using NumPy linear percentiles. P50, P99, P99−P50, and R=(P99−P50)/P50 are recomputed on the retained sample. Each execution remains a separate observation; means, sample standard deviations, and ranges summarize executions. Raw coverage and archive evidence remain intact. Per-run summaries record cutoffs and excluded source/input IDs.

The frozen selection.json records the historical decisions used to execute this matrix. selection_p1_p99.json records revised screening choices under the new policy; it does not change the actual A/B scenes of completed executions.

The first cell letter selects LiDAR input and the second selects camera input. LiDAR co-runner contrasts are AB−AA and BB−BA; camera co-runner contrasts are BA−AA and BB−AB. Own-input contrasts are compared with the corresponding single isolated A/B measurements. Isolation has one execution per scene/mode, so its uncertainty cannot be estimated from repetitions.

Observed associations are the same-scene screening differences. The crossed contrasts support input effects under the fixed replay/resource protocol; changes in processed-frame coverage can mediate these effects. Common-processed-frame comparisons are supplementary and condition on completion. They do not establish an internal GPU mechanism.

Expected bag records are not observed publications. Relay publication is the measured publication boundary. Missing bag-to-relay coverage is upstream missing coverage; published inputs absent from callbacks are dropped/overwritten without a queue-internal mechanism claim.

Inference NVTX ranges include recorded CUDA completion; decode and preprocessing use the same retained frame identities and remain separate. throughput_hz is the filtered inference count divided by the original replay-resume-to-max(window-end,last-completion) duration. observed_throughput_hz and completed_count retain actual completion evidence; elapsed and drain boundaries are unchanged.

Sequential frames are temporally dependent. P99 with <100 observations is especially fragile; <1,000 is a sparse-tail estimate. Unique-source counts and individual-run warnings are retained. Targeted longer common-window measurements are recommended for unstable tails; this study does not automatically add repetitions.

Kernel exports retain names, timing, process/context/stream IDs, launch correlations, and supported model/input/module attribution. Coverage includes unresolved kernel counts and summed GPU kernel duration (overlap is not collapsed). Depth-zero annotations cannot resolve individual layers; module/layer hooks with correlated launches would be needed. No layer-detail runs were scheduled.

MPS comparisons include only identical actual ordered scene combinations. Selection evidence reused in AA/BB is marked; selection-conditioned comparisons may be optimistic.

## 3DSSD and YOLOv3 assessed separately

Use the requested approximately 1 ms margin for absolute changes in P99−P50. Small observed effects do not establish statistical equivalence with one isolated execution per condition.

| Model | MPS | Effect | Contrast | Mean ΔR | SD | Range |
|---|---|---|---|---:|---:|---:|
| yolov3 | False | co_runner | BA-AA | 0.01058 | 0.009163084026641984 | 0.018292 |
| yolov3 | False | co_runner | BB-AB | -0.045423 | 0.014310146430442655 | 0.027779 |
| yolov3 | False | own_input | AB-AA | 0.091968 | 0.0036559549890422724 | 0.0071028 |
| yolov3 | False | own_input | BB-BA | 0.035966 | 0.017156430454482466 | 0.032014 |
| yolov3 | False | interaction | BB-BA-AB+AA | -0.056002 | 0.015218035472895306 | 0.029966 |
| 3dssd | False | co_runner | AB-AA | 0.0039171 | 0.00607024291284159 | 0.01121 |
| 3dssd | False | co_runner | BB-BA | 0.0053333 | 0.0069119055475395175 | 0.013747 |
| 3dssd | False | own_input | BA-AA | -0.001466 | 0.006602258921769894 | 0.013174 |
| 3dssd | False | own_input | BB-AB | -4.9835e-05 | 0.006467335982884537 | 0.01198 |
| 3dssd | False | interaction | BB-BA-AB+AA | 0.0014162 | 0.012500862662728637 | 0.023388 |
| yolov3 | True | co_runner | BA-AA | 0.023436 | 0.06555682724556783 | 0.12065 |
| yolov3 | True | co_runner | BB-AB | 0.031121 | 0.0685571535015953 | 0.11999 |
| yolov3 | True | own_input | AB-AA | 0.097731 | 0.0598022667551258 | 0.11496 |
| yolov3 | True | own_input | BB-BA | 0.10542 | 0.06727694164093681 | 0.12568 |
| yolov3 | True | interaction | BB-BA-AB+AA | 0.0076856 | 0.12055356758715985 | 0.24064 |
| 3dssd | True | co_runner | AB-AA | 0.012486 | 0.0024165635652364705 | 0.0043213 |
| 3dssd | True | co_runner | BB-BA | 0.0042067 | 0.0030866105493180503 | 0.0060311 |
| 3dssd | True | own_input | BA-AA | 0.0082068 | 0.007914734338225187 | 0.014585 |
| 3dssd | True | own_input | BB-AB | -7.2858e-05 | 0.0043761655236128315 | 0.0087508 |
| 3dssd | True | interaction | BB-BA-AB+AA | -0.0082797 | 0.005077764646818601 | 0.010066 |

## Remaining evidence blockers


## All pairs: execution-level input effects

| Pair | Model | MPS | Effect | Contrast | Mean ΔR | SD | Min | Max |
|---|---|---|---|---|---:|---:|---:|---:|
| 3dssd+yolov3 | yolov3 | False | co_runner | BA-AA | 0.01058 | 0.009163084026641984 | 0.0011116 | 0.019404 |
| 3dssd+yolov3 | yolov3 | False | co_runner | BB-AB | -0.045423 | 0.014310146430442655 | -0.061301 | -0.033521 |
| 3dssd+yolov3 | yolov3 | False | own_input | AB-AA | 0.091968 | 0.0036559549890422724 | 0.088918 | 0.096021 |
| 3dssd+yolov3 | yolov3 | False | own_input | BB-BA | 0.035966 | 0.017156430454482466 | 0.016394 | 0.048408 |
| 3dssd+yolov3 | yolov3 | False | interaction | BB-BA-AB+AA | -0.056002 | 0.015218035472895306 | -0.072524 | -0.042558 |
| 3dssd+yolov3 | 3dssd | False | co_runner | AB-AA | 0.0039171 | 0.00607024291284159 | -0.0030334 | 0.008177 |
| 3dssd+yolov3 | 3dssd | False | co_runner | BB-BA | 0.0053333 | 0.0069119055475395175 | -0.0011188 | 0.012628 |
| 3dssd+yolov3 | 3dssd | False | own_input | BA-AA | -0.001466 | 0.006602258921769894 | -0.0083131 | 0.0048606 |
| 3dssd+yolov3 | 3dssd | False | own_input | BB-AB | -4.9835e-05 | 0.006467335982884537 | -0.0046317 | 0.0073481 |
| 3dssd+yolov3 | 3dssd | False | interaction | BB-BA-AB+AA | 0.0014162 | 0.012500862662728637 | -0.0077265 | 0.015661 |
| centerpoint+dino | dino | False | co_runner | BA-AA | 0.3402 | 0.021442911494230162 | 0.32577 | 0.36484 |
| centerpoint+dino | dino | False | co_runner | BB-AB | 0.36084 | 0.06634379983019494 | 0.32226 | 0.43744 |
| centerpoint+dino | dino | False | own_input | AB-AA | -0.013383 | 0.06731746725610405 | -0.0911 | 0.026771 |
| centerpoint+dino | dino | False | own_input | BB-BA | 0.0072525 | 0.0223270012521853 | -0.018498 | 0.021214 |
| centerpoint+dino | dino | False | interaction | BB-BA-AB+AA | 0.020636 | 0.04506740415618977 | -0.0077294 | 0.072602 |
| centerpoint+dino | centerpoint | False | co_runner | AB-AA | 0.018719 | 0.08298272978631428 | -0.04803 | 0.11163 |
| centerpoint+dino | centerpoint | False | co_runner | BB-BA | -0.0036779 | 0.0269758680724278 | -0.034081 | 0.017391 |
| centerpoint+dino | centerpoint | False | own_input | BA-AA | -0.025812 | 0.03383883456597983 | -0.047254 | 0.013198 |
| centerpoint+dino | centerpoint | False | own_input | BB-AB | -0.048209 | 0.07806591155676464 | -0.13762 | 0.0064332 |
| centerpoint+dino | centerpoint | False | interaction | BB-BA-AB+AA | -0.022397 | 0.07405438035541712 | -0.094239 | 0.053687 |
| centerpoint+yolov3 | yolov3 | False | co_runner | BA-AA | 0.54551 | 0.4199969060375696 | 0.062188 | 0.82178 |
| centerpoint+yolov3 | yolov3 | False | co_runner | BB-AB | 0.080244 | 0.13599285304847503 | -0.014032 | 0.23614 |
| centerpoint+yolov3 | yolov3 | False | own_input | AB-AA | 0.3292 | 0.49862266617513445 | -0.24405 | 0.66233 |
| centerpoint+yolov3 | yolov3 | False | own_input | BB-BA | -0.13606 | 0.05729989301799423 | -0.17349 | -0.070097 |
| centerpoint+yolov3 | yolov3 | False | interaction | BB-BA-AB+AA | -0.46527 | 0.5559161540427351 | -0.83581 | 0.17395 |
| centerpoint+yolov3 | centerpoint | False | co_runner | AB-AA | 0.0087361 | 0.0170085186376291 | -0.010646 | 0.021175 |
| centerpoint+yolov3 | centerpoint | False | co_runner | BB-BA | 0.010155 | 0.013803945637473903 | -0.0056498 | 0.019849 |
| centerpoint+yolov3 | centerpoint | False | own_input | BA-AA | 0.099321 | 0.029602867882737086 | 0.067073 | 0.12526 |
| centerpoint+yolov3 | centerpoint | False | own_input | BB-AB | 0.10074 | 0.00475940140134697 | 0.097568 | 0.10621 |
| centerpoint+yolov3 | centerpoint | False | interaction | BB-BA-AB+AA | 0.0014185 | 0.028668839382942405 | -0.026825 | 0.030495 |
| pointpillars+vit-upernet | vit-upernet | False | co_runner | BA-AA | 0.084541 | 0.02874096204658307 | 0.053308 | 0.10987 |
| pointpillars+vit-upernet | vit-upernet | False | co_runner | BB-AB | -0.026151 | 0.04103981928443247 | -0.056169 | 0.020614 |
| pointpillars+vit-upernet | vit-upernet | False | own_input | AB-AA | 0.082135 | 0.031709948684325605 | 0.046822 | 0.10818 |
| pointpillars+vit-upernet | vit-upernet | False | own_input | BB-BA | -0.028557 | 0.038704383601174897 | -0.061366 | 0.014129 |
| pointpillars+vit-upernet | vit-upernet | False | interaction | BB-BA-AB+AA | -0.11069 | 0.06761861580547426 | -0.15277 | -0.032694 |
| pointpillars+vit-upernet | pointpillars | False | co_runner | AB-AA | 0.0018816 | 0.00510560972815472 | -0.0040008 | 0.0051628 |
| pointpillars+vit-upernet | pointpillars | False | co_runner | BB-BA | 0.0048017 | 0.00667301160438903 | -0.0024107 | 0.010756 |
| pointpillars+vit-upernet | pointpillars | False | own_input | BA-AA | -0.016295 | 0.005147277956205644 | -0.021202 | -0.010937 |
| pointpillars+vit-upernet | pointpillars | False | own_input | BB-AB | -0.013375 | 0.009349189626999549 | -0.023638 | -0.0053438 |
| pointpillars+vit-upernet | pointpillars | False | interaction | BB-BA-AB+AA | 0.0029201 | 0.00878733406096865 | -0.0068935 | 0.01006 |
| pointpillars+mask-rcnn | mask-rcnn | False | co_runner | BA-AA | -0.010201 | 0.018858042572059617 | -0.031904 | 0.0021921 |
| pointpillars+mask-rcnn | mask-rcnn | False | co_runner | BB-AB | -0.0057572 | 0.017528085459864562 | -0.022918 | 0.012116 |
| pointpillars+mask-rcnn | mask-rcnn | False | own_input | AB-AA | 0.042142 | 0.020450468056391582 | 0.019067 | 0.058026 |
| pointpillars+mask-rcnn | mask-rcnn | False | own_input | BB-BA | 0.046586 | 0.015283645703067176 | 0.032916 | 0.063087 |
| pointpillars+mask-rcnn | mask-rcnn | False | interaction | BB-BA-AB+AA | 0.0044443 | 0.035638075394509604 | -0.02511 | 0.04402 |
| pointpillars+mask-rcnn | pointpillars | False | co_runner | AB-AA | -0.0018349 | 0.009907594781212658 | -0.010585 | 0.0089225 |
| pointpillars+mask-rcnn | pointpillars | False | co_runner | BB-BA | 0.035645 | 0.008869235721459433 | 0.025505 | 0.041958 |
| pointpillars+mask-rcnn | pointpillars | False | own_input | BA-AA | 0.24964 | 0.04395839608749184 | 0.21917 | 0.30003 |
| pointpillars+mask-rcnn | pointpillars | False | own_input | BB-AB | 0.28712 | 0.056989929496025174 | 0.24852 | 0.35258 |
| pointpillars+mask-rcnn | pointpillars | False | interaction | BB-BA-AB+AA | 0.03748 | 0.013059192276780453 | 0.029346 | 0.052543 |
| 3dssd+yolov3 | yolov3 | True | co_runner | BA-AA | 0.023436 | 0.06555682724556783 | -0.022074 | 0.098577 |
| 3dssd+yolov3 | yolov3 | True | co_runner | BB-AB | 0.031121 | 0.0685571535015953 | -0.0097161 | 0.11027 |
| 3dssd+yolov3 | yolov3 | True | own_input | AB-AA | 0.097731 | 0.0598022667551258 | 0.049784 | 0.16474 |
| 3dssd+yolov3 | yolov3 | True | own_input | BB-BA | 0.10542 | 0.06727694164093681 | 0.056448 | 0.18213 |
| 3dssd+yolov3 | yolov3 | True | interaction | BB-BA-AB+AA | 0.0076856 | 0.12055356758715985 | -0.10829 | 0.13234 |
| 3dssd+yolov3 | 3dssd | True | co_runner | AB-AA | 0.012486 | 0.0024165635652364705 | 0.0097009 | 0.014022 |
| 3dssd+yolov3 | 3dssd | True | co_runner | BB-BA | 0.0042067 | 0.0030866105493180503 | 0.00081101 | 0.0068421 |
| 3dssd+yolov3 | 3dssd | True | own_input | BA-AA | 0.0082068 | 0.007914734338225187 | 0.0026905 | 0.017275 |
| 3dssd+yolov3 | 3dssd | True | own_input | BB-AB | -7.2858e-05 | 0.0043761655236128315 | -0.0044005 | 0.0043502 |
| 3dssd+yolov3 | 3dssd | True | interaction | BB-BA-AB+AA | -0.0082797 | 0.005077764646818601 | -0.012925 | -0.0028588 |
| centerpoint+dino | dino | True | co_runner | BA-AA | 0.23541 | 0.028392243925905298 | 0.21482 | 0.26779 |
| centerpoint+dino | dino | True | co_runner | BB-AB | 0.24202 | 0.030926392680042263 | 0.20869 | 0.2698 |
| centerpoint+dino | dino | True | own_input | AB-AA | -0.0083327 | 0.0034056441989176044 | -0.01121 | -0.0045726 |
| centerpoint+dino | dino | True | own_input | BB-BA | -0.0017196 | 0.02284552510962605 | -0.024128 | 0.02154 |
| centerpoint+dino | dino | True | interaction | BB-BA-AB+AA | 0.0066131 | 0.024163274844910407 | -0.014912 | 0.03275 |
| centerpoint+dino | centerpoint | True | co_runner | AB-AA | -0.021567 | 0.05383422502701278 | -0.070307 | 0.036216 |
| centerpoint+dino | centerpoint | True | co_runner | BB-BA | -0.0061331 | 0.03502146348196102 | -0.046293 | 0.018053 |
| centerpoint+dino | centerpoint | True | own_input | BA-AA | -0.027705 | 0.013992432783271296 | -0.040915 | -0.013043 |
| centerpoint+dino | centerpoint | True | own_input | BB-AB | -0.012271 | 0.024378262202655054 | -0.039418 | 0.0077495 |
| centerpoint+dino | centerpoint | True | interaction | BB-BA-AB+AA | 0.015434 | 0.03824833489247763 | -0.026375 | 0.048664 |
| centerpoint+yolov3 | yolov3 | True | co_runner | BA-AA | 1.5432 | 0.22870526095812746 | 1.2794 | 1.6858 |
| centerpoint+yolov3 | yolov3 | True | co_runner | BB-AB | 1.5116 | 0.20554059938367308 | 1.3182 | 1.7274 |
| centerpoint+yolov3 | yolov3 | True | own_input | AB-AA | -0.079324 | 0.29517371714638235 | -0.40722 | 0.1652 |
| centerpoint+yolov3 | yolov3 | True | own_input | BB-BA | -0.11089 | 0.13589313935305322 | -0.19736 | 0.045746 |
| centerpoint+yolov3 | yolov3 | True | interaction | BB-BA-AB+AA | -0.031564 | 0.285198508811214 | -0.34625 | 0.20986 |
| centerpoint+yolov3 | centerpoint | True | co_runner | AB-AA | 0.0062055 | 0.007543902389178824 | -0.00058741 | 0.014325 |
| centerpoint+yolov3 | centerpoint | True | co_runner | BB-BA | 0.0064299 | 0.010612671822502728 | -0.0057894 | 0.013343 |
| centerpoint+yolov3 | centerpoint | True | own_input | BA-AA | -0.12554 | 0.011107208357891656 | -0.13667 | -0.11445 |
| centerpoint+yolov3 | centerpoint | True | own_input | BB-AB | -0.12531 | 0.012141042325428614 | -0.13457 | -0.11157 |
| centerpoint+yolov3 | centerpoint | True | interaction | BB-BA-AB+AA | 0.00022445 | 0.017965083960548363 | -0.020114 | 0.01393 |
| pointpillars+vit-upernet | vit-upernet | True | co_runner | BA-AA | -0.066446 | 0.012723575045984734 | -0.081105 | -0.058269 |
| pointpillars+vit-upernet | vit-upernet | True | co_runner | BB-AB | -0.064624 | 0.011990617537812855 | -0.075798 | -0.051957 |
| pointpillars+vit-upernet | vit-upernet | True | own_input | AB-AA | 0.011237 | 0.004579801992469434 | 0.0076341 | 0.016391 |
| pointpillars+vit-upernet | vit-upernet | True | own_input | BB-BA | 0.013058 | 0.0028828841479104344 | 0.010236 | 0.015998 |
| pointpillars+vit-upernet | vit-upernet | True | interaction | BB-BA-AB+AA | 0.0018213 | 0.006926037937176094 | -0.0061551 | 0.0063124 |
| pointpillars+vit-upernet | pointpillars | True | co_runner | AB-AA | 0.013948 | 0.0238641967290893 | -0.0015144 | 0.041433 |
| pointpillars+vit-upernet | pointpillars | True | co_runner | BB-BA | 0.14015 | 0.0061197983099606 | 0.13314 | 0.14442 |
| pointpillars+vit-upernet | pointpillars | True | own_input | BA-AA | 0.050956 | 0.013308377813759117 | 0.03581 | 0.060781 |
| pointpillars+vit-upernet | pointpillars | True | own_input | BB-AB | 0.17716 | 0.024864593807898086 | 0.15249 | 0.20221 |
| pointpillars+vit-upernet | pointpillars | True | interaction | BB-BA-AB+AA | 0.1262 | 0.029977016805261873 | 0.091709 | 0.14594 |
| pointpillars+mask-rcnn | mask-rcnn | True | co_runner | BA-AA | -0.049397 | 0.012866908695897378 | -0.063026 | -0.037459 |
| pointpillars+mask-rcnn | mask-rcnn | True | co_runner | BB-AB | 0.030479 | 0.011858450484643866 | 0.016853 | 0.038465 |
| pointpillars+mask-rcnn | mask-rcnn | True | own_input | AB-AA | -0.11471 | 0.006077416847310777 | -0.1183 | -0.1077 |
| pointpillars+mask-rcnn | mask-rcnn | True | own_input | BB-BA | -0.034838 | 0.01729396835004001 | -0.053384 | -0.019153 |
| pointpillars+mask-rcnn | mask-rcnn | True | interaction | BB-BA-AB+AA | 0.079876 | 0.023069430924530787 | 0.054313 | 0.099145 |
| pointpillars+mask-rcnn | pointpillars | True | co_runner | AB-AA | 0.12817 | 0.01950892521897135 | 0.10635 | 0.14393 |
| pointpillars+mask-rcnn | pointpillars | True | co_runner | BB-BA | 0.19143 | 0.03368324094286254 | 0.17092 | 0.23031 |
| pointpillars+mask-rcnn | pointpillars | True | own_input | BA-AA | 0.01016 | 0.013674042596900174 | -0.0056115 | 0.018695 |
| pointpillars+mask-rcnn | pointpillars | True | own_input | BB-AB | 0.073425 | 0.03925940840435569 | 0.045684 | 0.11835 |
| pointpillars+mask-rcnn | pointpillars | True | interaction | BB-BA-AB+AA | 0.063265 | 0.05289474181857098 | 0.026989 | 0.12396 |
