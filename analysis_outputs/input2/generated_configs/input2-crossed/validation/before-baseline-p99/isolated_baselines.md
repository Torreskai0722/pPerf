# Single-model baseline inference times

Scenes are on the x-axis; CUDA-complete inference time in milliseconds is on the y-axis. Each model has separate MPS-off and MPS-on panels sharing a y-scale. Each violin contains all completed non-warmup observations from one isolated execution, without tail trimming. Decode and preprocessing are excluded from this inference range. Counts and unique source-frame counts appear under each scene; missing evidence is left empty. Within-execution frames are temporally dependent and are not independent repetitions.

[Download all models as a multipage PDF](plots/isolated/isolated_baselines.pdf)

## 3dssd

![3dssd baseline violins](plots/isolated/3dssd.png)

## centerpoint

![centerpoint baseline violins](plots/isolated/centerpoint.png)

## dino

![dino baseline violins](plots/isolated/dino.png)

## mask-rcnn

![mask-rcnn baseline violins](plots/isolated/mask-rcnn.png)

## pointpillars

![pointpillars baseline violins](plots/isolated/pointpillars.png)

## vit-upernet

![vit-upernet baseline violins](plots/isolated/vit-upernet.png)

## yolov3

![yolov3 baseline violins](plots/isolated/yolov3.png)
