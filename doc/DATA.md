# Data Directory Structure

This directory contains the data organization and setup instructions for perf_ws.

## Directory Structure

perf_ws expects your data and outputs to be organized as follows:

```
perf_ws/
  data/
    nuscenes/
      samples/
        CAM_FRONT/
        CAM_BACK/
        CAM_BACK_LEFT/
        CAM_BACK_RIGHT/
        CAM_FRONT_LEFT/
        CAM_FRONT_RIGHT/
        LIDAR_TOP/
        RADAR_BACK_LEFT/
        RADAR_BACK_RIGHT/
        RADAR_FRONT/
        RADAR_FRONT_LEFT/
        RADAR_FRONT_RIGHT/
      sweeps/
        CAM_FRONT/
        CAM_BACK/
        CAM_BACK_LEFT/
        CAM_BACK_RIGHT/
        CAM_FRONT_LEFT/
        CAM_FRONT_RIGHT/
        LIDAR_TOP/
        RADAR_BACK_LEFT/
        RADAR_BACK_RIGHT/
        RADAR_FRONT/
        RADAR_FRONT_LEFT/
        RADAR_FRONT_RIGHT/
      maps/
      v1.0-mini/
    bag/
      # ROS2 bag files for replay and benchmarking
      *.mcap
      ...
  outputs/
    # Inference results, logs, and profiling outputs
    lidar_pred_*.json
    image_pred_*.json
    delays_*.csv
    gpu_*.csv
    cpu_*.csv
    ...
  analysis_outputs/
    # Plots, figures, and processed analysis results
    *.png
    *.txt
    *.csv
    ...
```

## Data Setup Instructions

### 1. NuScenes Dataset

- **Download**: Get the NuScenes dataset from the [official website](https://www.nuscenes.org/)
- **Extract**: Place the extracted dataset under `data/nuscenes/`
- **Structure**: Ensure the standard NuScenes directory structure is maintained
- **Versions**: Supports both v1.0-mini and full v1.0-trainval versions

### 2. ROS2 Bag Files

- **Location**: Place ROS2 bag files (`.mcap` format) under `data/bag/`
- **Naming**: Use descriptive names that include scene information
- **Usage**: These files are used by `sensor_replayer.py` for benchmarking