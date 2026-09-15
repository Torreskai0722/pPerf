# Input2 scene-distribution results

The expanded matrix contains 216/216 valid runs: 3 LiDAR models × 6 image
models × 6 scenes × 2 MPS modes. The latency summary contains 432 rows, with
one row for each model in every cell. These are descriptive results from one
execution per cell; frame-level variation does not establish run-to-run
uncertainty, statistical significance, or causality.

Across matched pair/scene/model cells, enabling MPS changed the median latency
by a median of -23.50% for LiDAR models and -20.02% for image models. Individual
changes ranged from -41.30% to +9.52% for LiDAR and -77.88% to +6.38% for
images, so the direction was not universal. CenterPoint was generally most
sensitive to scene: its lowest median repeatedly occurred in scene-0245,
while scene-0184 frequently produced its highest median. The other LiDAR
models were usually less scene-sensitive, apart from PointPillars paired with
ViT-UPerNet.

The table reports each target modality's minimum-to-maximum median latency
across the six active scenes, in milliseconds. The scene suffix producing each
endpoint is shown in parentheses.

| model pair | off: LiDAR | off: image | on: LiDAR | on: image |
|---|---:|---:|---:|---:|
| 3dssd+detr | 64.22 (0434)–65.89 (0184) | 44.45 (0738)–46.20 (0245) | 48.06 (0434)–48.95 (0245) | 24.93 (0738)–26.61 (0245) |
| 3dssd+dino | 75.31 (0434)–78.17 (0398) | 132.99 (0738)–136.85 (0770) | 80.26 (0245)–83.24 (0184) | 78.16 (0245)–79.69 (0184) |
| 3dssd+faster-rcnn | 70.56 (0434)–73.76 (0770) | 63.73 (0184)–69.81 (0738) | 49.53 (0245)–51.02 (0184) | 39.64 (0434)–40.19 (0184) |
| 3dssd+mask-rcnn | 71.35 (0434)–77.88 (0245) | 71.60 (0434)–90.29 (0245) | 50.82 (0184)–62.61 (0245) | 42.16 (0184)–55.01 (0245) |
| 3dssd+vit-upernet | 76.89 (0434)–79.58 (0398) | 60.82 (0434)–68.50 (0738) | 50.71 (0434)–51.15 (0770) | 38.87 (0434)–40.65 (0398) |
| 3dssd+yolov3 | 47.53 (0770)–48.56 (0184) | 33.11 (0434)–46.16 (0770) | 42.00 (0398)–43.10 (0434) | 10.21 (0770)–10.92 (0434) |
| centerpoint+detr | 164.25 (0245)–229.13 (0184) | 42.63 (0245)–49.52 (0770) | 133.70 (0245)–182.69 (0184) | 27.48 (0770)–41.18 (0184) |
| centerpoint+dino | 246.86 (0245)–346.28 (0184) | 95.82 (0434)–103.67 (0245) | 186.33 (0245)–262.76 (0184) | 87.98 (0245)–92.93 (0184) |
| centerpoint+faster-rcnn | 207.55 (0245)–265.43 (0184) | 59.77 (0434)–65.45 (0770) | 149.52 (0245)–195.69 (0184) | 41.68 (0770)–52.98 (0738) |
| centerpoint+mask-rcnn | 239.60 (0245)–274.17 (0770) | 60.67 (0434)–72.89 (0398) | 160.26 (0245)–219.20 (0770) | 49.24 (0184)–63.07 (0245) |
| centerpoint+vit-upernet | 224.31 (0245)–301.49 (0184) | 56.03 (0245)–60.67 (0184) | 148.13 (0245)–196.87 (0184) | 42.81 (0770)–49.31 (0434) |
| centerpoint+yolov3 | 124.42 (0245)–170.66 (0184) | 24.50 (0245)–34.71 (0770) | 110.14 (0245)–157.23 (0184) | 11.44 (0738)–16.70 (0398) |
| pointpillars+detr | 51.48 (0245)–54.77 (0184) | 29.60 (0434)–37.68 (0738) | 40.63 (0245)–44.31 (0738) | 25.89 (0434)–29.08 (0738) |
| pointpillars+dino | 95.42 (0434)–98.48 (0738) | 91.46 (0434)–94.35 (0738) | 61.41 (0245)–67.82 (0184) | 80.30 (0245)–83.39 (0738) |
| pointpillars+faster-rcnn | 60.89 (0245)–64.58 (0738) | 47.55 (0398)–52.03 (0434) | 47.38 (0245)–54.25 (0738) | 41.93 (0245)–48.92 (0738) |
| pointpillars+mask-rcnn | 66.33 (0434)–73.94 (0398) | 56.00 (0184)–67.04 (0245) | 49.17 (0245)–53.79 (0738) | 48.50 (0184)–57.36 (0245) |
| pointpillars+vit-upernet | 51.60 (0770)–68.84 (0184) | 42.43 (0245)–49.78 (0770) | 36.21 (0245)–52.27 (0738) | 39.97 (0245)–46.82 (0738) |
| pointpillars+yolov3 | 35.09 (0245)–38.59 (0738) | 12.28 (0245)–15.06 (0738) | 30.45 (0245)–33.41 (0184) | 11.82 (0398)–12.92 (0738) |

The violin plots add the prior scene-0252 clear reference to these six active
scenes for every model pair and MPS mode. ECDFs and heatmaps use only the six
active input2 scenes.
