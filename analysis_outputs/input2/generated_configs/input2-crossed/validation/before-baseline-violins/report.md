# Controlled environment sensitivity

Validated analysis rows: 376. Blocked execution slots: 0.

R=(P99−P50)/P50 uses NumPy linear percentiles with no trimming. Each execution remains a separate observation; means, sample standard deviations, and ranges summarize executions, never pooled frames.

The first cell letter selects LiDAR input and the second selects camera input. LiDAR co-runner contrasts are AB−AA and BB−BA; camera co-runner contrasts are BA−AA and BB−AB. Own-input contrasts are compared with the corresponding single isolated A/B measurements. Isolation has one execution per scene/mode, so its uncertainty cannot be estimated from repetitions.

Observed associations are the same-scene screening differences. The crossed contrasts support input effects under the fixed replay/resource protocol; changes in processed-frame coverage can mediate these effects. Common-processed-frame comparisons are supplementary and condition on completion. They do not establish an internal GPU mechanism.

Expected bag records are not observed publications. Relay publication is the measured publication boundary. Missing bag-to-relay coverage is upstream missing coverage; published inputs absent from callbacks are dropped/overwritten without a queue-internal mechanism claim.

Inference NVTX ranges include recorded CUDA completion; decode and preprocessing remain separate. Throughput uses completed count divided by time from replay resume to max(common-window end, last model completion), with elapsed and drain durations retained.

Sequential frames are temporally dependent. P99 with <100 observations is especially fragile; <1,000 is a sparse-tail estimate. Unique-source counts and individual-run warnings are retained. Targeted longer common-window measurements are recommended for unstable tails; this study does not automatically add repetitions.

Kernel exports retain names, timing, process/context/stream IDs, launch correlations, and supported model/input/module attribution. Coverage includes unresolved kernel counts and summed GPU kernel duration (overlap is not collapsed). Depth-zero annotations cannot resolve individual layers; module/layer hooks with correlated launches would be needed. No layer-detail runs were scheduled.

MPS comparisons include only identical actual ordered scene combinations. Selection evidence reused in AA/BB is marked; selection-conditioned comparisons may be optimistic.

## 3DSSD and YOLOv3 assessed separately

No insensitivity/equivalence margin was specified. Small measured effects alone do not establish an insensitive control.

| Model | MPS | Effect | Contrast | Mean ΔR | SD | Range |
|---|---|---|---|---:|---:|---:|
| yolov3 | False | co_runner | BA-AA | 0.010441 | 0.005826050215688002 | 0.011651 |
| yolov3 | False | co_runner | BB-AB | -0.041408 | 0.009076203496464035 | 0.018055 |
| yolov3 | False | own_input | AB-AA | 0.092811 | 0.007114429370870979 | 0.013324 |
| yolov3 | False | own_input | BB-BA | 0.040962 | 0.01282219033008125 | 0.025449 |
| yolov3 | False | interaction | BB-BA-AB+AA | -0.051849 | 0.007305348277577922 | 0.013123 |
| 3dssd | False | co_runner | AB-AA | 0.0063177 | 0.004014580895989894 | 0.00795 |
| 3dssd | False | co_runner | BB-BA | 0.00016141 | 0.012464895056059027 | 0.024276 |
| 3dssd | False | own_input | BA-AA | 0.0093082 | 0.012479355290735742 | 0.023728 |
| 3dssd | False | own_input | BB-AB | 0.0031519 | 0.007469924233884728 | 0.01489 |
| 3dssd | False | interaction | BB-BA-AB+AA | -0.0061563 | 0.016272564429764918 | 0.032226 |
| yolov3 | True | co_runner | BA-AA | 0.023297 | 0.10463355538224263 | 0.18686 |
| yolov3 | True | co_runner | BB-AB | 0.022221 | 0.11502267740832521 | 0.22841 |
| yolov3 | True | own_input | AB-AA | 0.081957 | 0.10543403267341568 | 0.18813 |
| yolov3 | True | own_input | BB-BA | 0.080881 | 0.1194026741978874 | 0.23872 |
| yolov3 | True | interaction | BB-BA-AB+AA | -0.0010756 | 0.21635562650493137 | 0.41527 |
| 3dssd | True | co_runner | AB-AA | 0.015833 | 0.004661061194689981 | 0.0086246 |
| 3dssd | True | co_runner | BB-BA | -0.0063219 | 0.00820965063235419 | 0.016242 |
| 3dssd | True | own_input | BA-AA | 0.01754 | 0.0023044253360599404 | 0.0045659 |
| 3dssd | True | own_input | BB-AB | -0.0046148 | 0.0030562688797617283 | 0.0058786 |
| 3dssd | True | interaction | BB-BA-AB+AA | -0.022154 | 0.0048308490494579975 | 0.0089556 |

## Remaining evidence blockers


## All pairs: execution-level input effects

| Pair | Model | MPS | Effect | Contrast | Mean ΔR | SD | Min | Max |
|---|---|---|---|---|---:|---:|---:|---:|
| 3dssd+yolov3 | yolov3 | False | co_runner | BA-AA | 0.010441 | 0.005826050215688002 | 0.0046501 | 0.016302 |
| 3dssd+yolov3 | yolov3 | False | co_runner | BB-AB | -0.041408 | 0.009076203496464035 | -0.049894 | -0.031838 |
| 3dssd+yolov3 | yolov3 | False | own_input | AB-AA | 0.092811 | 0.007114429370870979 | 0.087591 | 0.10091 |
| 3dssd+yolov3 | yolov3 | False | own_input | BB-BA | 0.040962 | 0.01282219033008125 | 0.027326 | 0.052775 |
| 3dssd+yolov3 | yolov3 | False | interaction | BB-BA-AB+AA | -0.051849 | 0.007305348277577922 | -0.060265 | -0.047142 |
| 3dssd+yolov3 | 3dssd | False | co_runner | AB-AA | 0.0063177 | 0.004014580895989894 | 0.0026674 | 0.010617 |
| 3dssd+yolov3 | 3dssd | False | co_runner | BB-BA | 0.00016141 | 0.012464895056059027 | -0.010339 | 0.013937 |
| 3dssd+yolov3 | 3dssd | False | own_input | BA-AA | 0.0093082 | 0.012479355290735742 | -0.00032137 | 0.023407 |
| 3dssd+yolov3 | 3dssd | False | own_input | BB-AB | 0.0031519 | 0.007469924233884728 | -0.0039423 | 0.010948 |
| 3dssd+yolov3 | 3dssd | False | interaction | BB-BA-AB+AA | -0.0061563 | 0.016272564429764918 | -0.020957 | 0.011269 |
| centerpoint+dino | dino | False | co_runner | BA-AA | 0.30344 | 0.025023370284752534 | 0.28089 | 0.33036 |
| centerpoint+dino | dino | False | co_runner | BB-AB | 0.33942 | 0.05249779171034572 | 0.30519 | 0.39986 |
| centerpoint+dino | dino | False | own_input | AB-AA | -0.029418 | 0.04862344408055096 | -0.085391 | 0.0023766 |
| centerpoint+dino | dino | False | own_input | BB-BA | 0.0065542 | 0.02137998733876654 | -0.015893 | 0.026678 |
| centerpoint+dino | dino | False | interaction | BB-BA-AB+AA | 0.035972 | 0.02947779385441989 | 0.014117 | 0.069499 |
| centerpoint+dino | centerpoint | False | co_runner | AB-AA | -0.019016 | 0.11295417257720344 | -0.13069 | 0.095176 |
| centerpoint+dino | centerpoint | False | co_runner | BB-BA | -0.0019614 | 0.02298571798650305 | -0.02724 | 0.017685 |
| centerpoint+dino | centerpoint | False | own_input | BA-AA | -0.071859 | 0.052417429795613736 | -0.12691 | -0.022546 |
| centerpoint+dino | centerpoint | False | own_input | BB-AB | -0.054804 | 0.07895474896957248 | -0.14361 | 0.007452 |
| centerpoint+dino | centerpoint | False | interaction | BB-BA-AB+AA | 0.017055 | 0.10774426735111942 | -0.077491 | 0.13436 |
| centerpoint+yolov3 | yolov3 | False | co_runner | BA-AA | 0.54888 | 0.4213382639951492 | 0.064307 | 0.82886 |
| centerpoint+yolov3 | yolov3 | False | co_runner | BB-AB | 0.08022 | 0.1318740936765096 | -0.010344 | 0.23152 |
| centerpoint+yolov3 | yolov3 | False | own_input | AB-AA | 0.33 | 0.4961144204284708 | -0.24013 | 0.66349 |
| centerpoint+yolov3 | yolov3 | False | own_input | BB-BA | -0.13865 | 0.057084842604921755 | -0.17571 | -0.072914 |
| centerpoint+yolov3 | yolov3 | False | interaction | BB-BA-AB+AA | -0.46866 | 0.5531840091705388 | -0.8392 | 0.16721 |
| centerpoint+yolov3 | centerpoint | False | co_runner | AB-AA | 0.0056601 | 0.01807662736063314 | -0.013939 | 0.021678 |
| centerpoint+yolov3 | centerpoint | False | co_runner | BB-BA | -0.0010639 | 0.020191020350346076 | -0.019226 | 0.020677 |
| centerpoint+yolov3 | centerpoint | False | own_input | BA-AA | 0.1131 | 0.019922267874128576 | 0.096147 | 0.13504 |
| centerpoint+yolov3 | centerpoint | False | own_input | BB-AB | 0.10637 | 0.01272810737440893 | 0.094137 | 0.11954 |
| centerpoint+yolov3 | centerpoint | False | interaction | BB-BA-AB+AA | -0.006724 | 0.029620586216924483 | -0.040905 | 0.011437 |
| pointpillars+vit-upernet | vit-upernet | False | co_runner | BA-AA | 0.08427 | 0.021122435476583136 | 0.062449 | 0.10462 |
| pointpillars+vit-upernet | vit-upernet | False | co_runner | BB-AB | -0.02688 | 0.05012310031029925 | -0.067133 | 0.029262 |
| pointpillars+vit-upernet | vit-upernet | False | own_input | AB-AA | 0.089978 | 0.022204354831571568 | 0.064849 | 0.10695 |
| pointpillars+vit-upernet | vit-upernet | False | own_input | BB-BA | -0.021172 | 0.04578565075254681 | -0.049251 | 0.031662 |
| pointpillars+vit-upernet | vit-upernet | False | interaction | BB-BA-AB+AA | -0.11115 | 0.06757362602695732 | -0.15288 | -0.033187 |
| pointpillars+vit-upernet | pointpillars | False | co_runner | AB-AA | -0.0003064 | 0.005244341164226269 | -0.0037817 | 0.005726 |
| pointpillars+vit-upernet | pointpillars | False | co_runner | BB-BA | 0.0011809 | 0.011354063981064364 | -0.010909 | 0.011619 |
| pointpillars+vit-upernet | pointpillars | False | own_input | BA-AA | -0.01151 | 0.0038484328405403254 | -0.015914 | -0.008793 |
| pointpillars+vit-upernet | pointpillars | False | own_input | BB-AB | -0.010023 | 0.00599718030724201 | -0.01592 | -0.0039303 |
| pointpillars+vit-upernet | pointpillars | False | interaction | BB-BA-AB+AA | 0.0014873 | 0.007460761233676249 | -0.0071269 | 0.0058926 |
| pointpillars+mask-rcnn | mask-rcnn | False | co_runner | BA-AA | -0.0085855 | 0.01371339434958667 | -0.024362 | 0.00048368 |
| pointpillars+mask-rcnn | mask-rcnn | False | co_runner | BB-AB | -0.015771 | 0.016025618695842285 | -0.033786 | -0.0031001 |
| pointpillars+mask-rcnn | mask-rcnn | False | own_input | AB-AA | 0.051819 | 0.009928879231636174 | 0.042317 | 0.062126 |
| pointpillars+mask-rcnn | mask-rcnn | False | own_input | BB-BA | 0.044633 | 0.01323958663297941 | 0.030218 | 0.05625 |
| pointpillars+mask-rcnn | mask-rcnn | False | interaction | BB-BA-AB+AA | -0.0071858 | 0.02313174292721095 | -0.031907 | 0.013934 |
| pointpillars+mask-rcnn | pointpillars | False | co_runner | AB-AA | -0.0085028 | 0.017926801108298605 | -0.028492 | 0.0061496 |
| pointpillars+mask-rcnn | pointpillars | False | co_runner | BB-BA | 0.085142 | 0.04000210404778243 | 0.039867 | 0.1157 |
| pointpillars+mask-rcnn | pointpillars | False | own_input | BA-AA | 0.28201 | 0.057436535191742244 | 0.22416 | 0.33902 |
| pointpillars+mask-rcnn | pointpillars | False | own_input | BB-AB | 0.37565 | 0.0068950626597124075 | 0.36835 | 0.38206 |
| pointpillars+mask-rcnn | pointpillars | False | interaction | BB-BA-AB+AA | 0.093645 | 0.05058112693535921 | 0.043032 | 0.14419 |
| 3dssd+yolov3 | yolov3 | True | co_runner | BA-AA | 0.023297 | 0.10463355538224263 | -0.042941 | 0.14392 |
| 3dssd+yolov3 | yolov3 | True | co_runner | BB-AB | 0.022221 | 0.11502267740832521 | -0.099894 | 0.12851 |
| 3dssd+yolov3 | yolov3 | True | own_input | AB-AA | 0.081957 | 0.10543403267341568 | 0.015384 | 0.20352 |
| 3dssd+yolov3 | yolov3 | True | own_input | BB-BA | 0.080881 | 0.1194026741978874 | -0.040299 | 0.19842 |
| 3dssd+yolov3 | yolov3 | True | interaction | BB-BA-AB+AA | -0.0010756 | 0.21635562650493137 | -0.24382 | 0.17145 |
| 3dssd+yolov3 | 3dssd | True | co_runner | AB-AA | 0.015833 | 0.004661061194689981 | 0.012542 | 0.021166 |
| 3dssd+yolov3 | 3dssd | True | co_runner | BB-BA | -0.0063219 | 0.00820965063235419 | -0.015137 | 0.001105 |
| 3dssd+yolov3 | 3dssd | True | own_input | BA-AA | 0.01754 | 0.0023044253360599404 | 0.015075 | 0.019641 |
| 3dssd+yolov3 | 3dssd | True | own_input | BB-AB | -0.0046148 | 0.0030562688797617283 | -0.0080376 | -0.002159 |
| 3dssd+yolov3 | 3dssd | True | interaction | BB-BA-AB+AA | -0.022154 | 0.0048308490494579975 | -0.027679 | -0.018723 |
| centerpoint+dino | dino | True | co_runner | BA-AA | 0.21164 | 0.04109571009832647 | 0.18361 | 0.25882 |
| centerpoint+dino | dino | True | co_runner | BB-AB | 0.23553 | 0.035958468008582906 | 0.19944 | 0.27135 |
| centerpoint+dino | dino | True | own_input | AB-AA | -0.025834 | 0.022642160173317744 | -0.04252 | -5.9832e-05 |
| centerpoint+dino | dino | True | own_input | BB-BA | -0.0019406 | 0.021530138227714664 | -0.026689 | 0.012478 |
| centerpoint+dino | dino | True | interaction | BB-BA-AB+AA | 0.023894 | 0.016897446703507086 | 0.012537 | 0.043312 |
| centerpoint+dino | centerpoint | True | co_runner | AB-AA | -0.00629 | 0.003994091549633876 | -0.010893 | -0.0037401 |
| centerpoint+dino | centerpoint | True | co_runner | BB-BA | -0.0079985 | 0.0362856710932422 | -0.04975 | 0.015917 |
| centerpoint+dino | centerpoint | True | own_input | BA-AA | -0.053444 | 0.0023515807960534743 | -0.056079 | -0.051557 |
| centerpoint+dino | centerpoint | True | own_input | BB-AB | -0.055153 | 0.040854624721121224 | -0.10159 | -0.024747 |
| centerpoint+dino | centerpoint | True | interaction | BB-BA-AB+AA | -0.0017085 | 0.03850882259859314 | -0.045514 | 0.02681 |
| centerpoint+yolov3 | yolov3 | True | co_runner | BA-AA | 1.4539 | 0.2451822292185793 | 1.1708 | 1.5962 |
| centerpoint+yolov3 | yolov3 | True | co_runner | BB-AB | 1.4531 | 0.3976271962710244 | 0.99436 | 1.6998 |
| centerpoint+yolov3 | yolov3 | True | own_input | AB-AA | 0.0056958 | 0.48597045724736215 | -0.472 | 0.49953 |
| centerpoint+yolov3 | yolov3 | True | own_input | BB-BA | 0.0048101 | 0.09153551125616176 | -0.10088 | 0.058375 |
| centerpoint+yolov3 | yolov3 | True | interaction | BB-BA-AB+AA | -0.00088574 | 0.5678971476298299 | -0.60042 | 0.52894 |
| centerpoint+yolov3 | centerpoint | True | co_runner | AB-AA | 0.0086758 | 0.007728909813913095 | 0.0032599 | 0.017527 |
| centerpoint+yolov3 | centerpoint | True | co_runner | BB-BA | 0.0088932 | 0.012688560859443102 | -0.0042447 | 0.021079 |
| centerpoint+yolov3 | centerpoint | True | own_input | BA-AA | -0.12846 | 0.006003078257932784 | -0.13398 | -0.12207 |
| centerpoint+yolov3 | centerpoint | True | own_input | BB-AB | -0.12824 | 0.014170885180325448 | -0.14384 | -0.11616 |
| centerpoint+yolov3 | centerpoint | True | interaction | BB-BA-AB+AA | 0.00021743 | 0.02015655430504857 | -0.021772 | 0.017819 |
| pointpillars+vit-upernet | vit-upernet | True | co_runner | BA-AA | -0.075618 | 0.004698257658194251 | -0.08044 | -0.071055 |
| pointpillars+vit-upernet | vit-upernet | True | co_runner | BB-AB | -0.062045 | 0.00790030859914138 | -0.069315 | -0.053638 |
| pointpillars+vit-upernet | vit-upernet | True | own_input | AB-AA | -0.002665 | 0.007450116280920836 | -0.010546 | 0.0042629 |
| pointpillars+vit-upernet | vit-upernet | True | own_input | BB-BA | 0.010908 | 0.013658459774299758 | -0.0045012 | 0.021522 |
| pointpillars+vit-upernet | vit-upernet | True | interaction | BB-BA-AB+AA | 0.013573 | 0.0065207522205076255 | 0.0060445 | 0.017417 |
| pointpillars+vit-upernet | pointpillars | True | co_runner | AB-AA | 0.016467 | 0.047457796426894426 | -0.037291 | 0.052556 |
| pointpillars+vit-upernet | pointpillars | True | co_runner | BB-BA | 0.13137 | 0.021453881088542247 | 0.10905 | 0.15183 |
| pointpillars+vit-upernet | pointpillars | True | own_input | BA-AA | 0.055125 | 0.04139397431354127 | 0.0073278 | 0.079211 |
| pointpillars+vit-upernet | pointpillars | True | own_input | BB-AB | 0.17003 | 0.031160453768699484 | 0.1357 | 0.19653 |
| pointpillars+vit-upernet | pointpillars | True | interaction | BB-BA-AB+AA | 0.1149 | 0.05706282754741781 | 0.05649 | 0.17051 |
| pointpillars+mask-rcnn | mask-rcnn | True | co_runner | BA-AA | -0.057792 | 0.025796175765540327 | -0.087552 | -0.041817 |
| pointpillars+mask-rcnn | mask-rcnn | True | co_runner | BB-AB | 0.039966 | 0.010443496825561354 | 0.028328 | 0.048521 |
| pointpillars+mask-rcnn | mask-rcnn | True | own_input | AB-AA | -0.13159 | 0.028552256651119146 | -0.16259 | -0.10637 |
| pointpillars+mask-rcnn | mask-rcnn | True | own_input | BB-BA | -0.033827 | 0.006458459670089238 | -0.038746 | -0.026513 |
| pointpillars+mask-rcnn | mask-rcnn | True | interaction | BB-BA-AB+AA | 0.097758 | 0.034241897167936276 | 0.070145 | 0.13607 |
| pointpillars+mask-rcnn | pointpillars | True | co_runner | AB-AA | 0.12168 | 0.016892724598753732 | 0.10248 | 0.13423 |
| pointpillars+mask-rcnn | pointpillars | True | co_runner | BB-BA | 0.19535 | 0.027152969432319787 | 0.17771 | 0.22662 |
| pointpillars+mask-rcnn | pointpillars | True | own_input | BA-AA | 0.0061809 | 0.008555436812424735 | -0.0036326 | 0.012071 |
| pointpillars+mask-rcnn | pointpillars | True | own_input | BB-AB | 0.079845 | 0.035709944080713825 | 0.053587 | 0.12051 |
| pointpillars+mask-rcnn | pointpillars | True | interaction | BB-BA-AB+AA | 0.073664 | 0.04399197443811357 | 0.043484 | 0.12414 |
