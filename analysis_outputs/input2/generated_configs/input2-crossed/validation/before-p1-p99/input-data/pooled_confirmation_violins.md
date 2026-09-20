# Pooled inference-time violin plots

Each violin treats the three executions of one confirmation condition as one combined sample. Every completed non-warmup inference is included once, with no tail trimming. Repeated source frames across executions retain their separate latency observations. Models, MPS modes, and actual input-scene combinations remain separate. Inference time includes CUDA completion; external decode/preprocessing are excluded. These are pooled descriptive distributions, not estimates of between-execution uncertainty.

[All plots as a multipage PDF](plots/pooled_confirmation/pooled_inference_times.pdf) · [Sample counts and source executions](pooled_inference_counts.csv)

## 3dssd+yolov3-0-3dssd

![Pooled inference times](plots/pooled_confirmation/3dssd+yolov3-0-3dssd.png)

## 3dssd+yolov3-0-yolov3

![Pooled inference times](plots/pooled_confirmation/3dssd+yolov3-0-yolov3.png)

## 3dssd+yolov3-1-3dssd

![Pooled inference times](plots/pooled_confirmation/3dssd+yolov3-1-3dssd.png)

## 3dssd+yolov3-1-yolov3

![Pooled inference times](plots/pooled_confirmation/3dssd+yolov3-1-yolov3.png)

## centerpoint+dino-0-centerpoint

![Pooled inference times](plots/pooled_confirmation/centerpoint+dino-0-centerpoint.png)

## centerpoint+dino-0-dino

![Pooled inference times](plots/pooled_confirmation/centerpoint+dino-0-dino.png)

## centerpoint+dino-1-centerpoint

![Pooled inference times](plots/pooled_confirmation/centerpoint+dino-1-centerpoint.png)

## centerpoint+dino-1-dino

![Pooled inference times](plots/pooled_confirmation/centerpoint+dino-1-dino.png)

## centerpoint+yolov3-0-centerpoint

![Pooled inference times](plots/pooled_confirmation/centerpoint+yolov3-0-centerpoint.png)

## centerpoint+yolov3-0-yolov3

![Pooled inference times](plots/pooled_confirmation/centerpoint+yolov3-0-yolov3.png)

## centerpoint+yolov3-1-centerpoint

![Pooled inference times](plots/pooled_confirmation/centerpoint+yolov3-1-centerpoint.png)

## centerpoint+yolov3-1-yolov3

![Pooled inference times](plots/pooled_confirmation/centerpoint+yolov3-1-yolov3.png)

## pointpillars+mask-rcnn-0-mask-rcnn

![Pooled inference times](plots/pooled_confirmation/pointpillars+mask-rcnn-0-mask-rcnn.png)

## pointpillars+mask-rcnn-0-pointpillars

![Pooled inference times](plots/pooled_confirmation/pointpillars+mask-rcnn-0-pointpillars.png)

## pointpillars+mask-rcnn-1-mask-rcnn

![Pooled inference times](plots/pooled_confirmation/pointpillars+mask-rcnn-1-mask-rcnn.png)

## pointpillars+mask-rcnn-1-pointpillars

![Pooled inference times](plots/pooled_confirmation/pointpillars+mask-rcnn-1-pointpillars.png)

## pointpillars+vit-upernet-0-pointpillars

![Pooled inference times](plots/pooled_confirmation/pointpillars+vit-upernet-0-pointpillars.png)

## pointpillars+vit-upernet-0-vit-upernet

![Pooled inference times](plots/pooled_confirmation/pointpillars+vit-upernet-0-vit-upernet.png)

## pointpillars+vit-upernet-1-pointpillars

![Pooled inference times](plots/pooled_confirmation/pointpillars+vit-upernet-1-pointpillars.png)

## pointpillars+vit-upernet-1-vit-upernet

![Pooled inference times](plots/pooled_confirmation/pointpillars+vit-upernet-1-vit-upernet.png)
