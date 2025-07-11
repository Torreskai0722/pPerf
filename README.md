# Multi-Tenant DNN Inference Profiling for Autonomous Driving

perf_ws is a suite of tools for profiling, benchmarking, and analyzing multi-model DNN inference pipelines for autonomous driving. It supports both LiDAR and camera (image) modalities, and is designed for use with the NuScenes dataset. perf_ws helps you understand, debug, and optimize inference pipelines running on CPUs and GPUs.

---

## Features

perf_ws offers a number of tools to analyze and visualize the performance of your models across multiple GPU and data streams. Some of the key features include:

ADD A GRAPH (FROM PAPER)

- **Profiling & Monitoring:**  
  - Fine-grained timing and call-depth profiling of DNN models (`pPerf.py`)
  - GPU and CPU utilization monitoring (NVML, psutil)
  - NVTX marker insertion for CUDA profiling

- **Flexible Inference Pipelines:**  
  - Single-model and multi-model (multi-threaded) inference nodes
  - Support for LiDAR, image, and multi-modal (BEVFusion) models
  - Real-time ROS2-based streaming and offline bag replay

- **Data Publishing & Replay:**  
  - Publish preloaded NuScenes data as ROS2 messages (`sensor_publisher.py`)
  - Replay recorded ROS2 bag files for offline benchmarking (`sensor_replayer.py`)

- **Comprehensive Logging:**  
  - Detailed timing logs (communication, decode, inference, end-to-end)
  - Output predictions and resource usage to JSON/CSV

- **Demos & Example Scripts:**
  - Ready-to-run experiment scripts for common scenarios and benchmarking (`experiment_scripts/`)

- **Post-Processing & Analysis:**
  - Tools for analyzing layer-wise, kernel-level, and end-to-end performance (`post_processing/`)
  - Visualization and reporting utilities for in-depth analysis

---

## Demo

**First time user?**  
Check out the [Demos & Example Scripts](#demos--example-scripts) below to get started quickly.

---

## Installation

The easiest way to get started with perf_ws is using Docker. We provide a pre-configured Docker environment with all dependencies installed. Our perf is tested on the following version ....

ADD IMPORTANT LIBRARY VERSIONS (CUDA Driver, Torch, etc.)

### Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t perf_ws -f Docker/pPerf.Dockerfile .
   ```

2. **Run the container:**
   ```bash
   docker run -it \
    --gpus all \
    --privileged \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --network=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v ~/pdnn/pPerf:/mmdetection3d_ros2 \
    -v /mnt/nas/Nuscenes:/mnt/nas/Nuscenes \
    --name pPerf-1 \
    pdnn-1:latest bash

   ```

3. **Inside the container, build the ROS2 workspace:**
   ```bash
   cd /mmdetection3d_ros2/perf_ws
   colcon build
   source install/setup.bash

---

## Quick Start

1. **Install dependencies**  
   Make sure you have all prerequisites installed (see above).

2. **Prepare your data**  
   Download and extract the NuScenes dataset. Update paths in your config if needed.

### Data Directory Structure

perf_ws expects your data and outputs to be organized as follows:

```
perf_ws/
  data/
    nuscenes/
      samples/
        CAM_FRONT/
        CAM_BACK/
        LIDAR_TOP/
        ...
      sweeps/
        CAM_FRONT/
        LIDAR_TOP/
        ...
      maps/
      ...
    bag/
      # ROS2 bag files for replay and benchmarking
      *.mcap
      ...
  outputs/
    # Inference results, logs, and profiling outputs
    lidar_pred_*.json
    image_pred_*.json
    delays_*.csv
    ...
```

- Place the NuScenes dataset under `data/nuscenes/` (with its standard structure).
- Place ROS2 bag files for replay and benchmarking under `data/bag/`.
- All experiment outputs, logs, and profiling results will be saved under `outputs/`.

4. **Launch the pipeline**

   - **Start the data publisher (reading from file directly):**
     ```bash
     ros2 run p_perf sensor_publisher.py
     ```

   - **(Optional) Replay a bag file:**
     ```bash
     ros2 run p_perf sensor_replayer.py
     ```

   - **Start the inference node (launch from single or multi-process):**
     ```bash
     ros2 run p_perf inferencer.py
     # or
     ros2 run p_perf inferencer_ms.py
     ```



5. **View results**  
   - Profiling outputs (timing, GPU/CPU stats) are saved in your specified data directory.
   - Analyze logs and outputs for performance, accuracy, and resource usage.

---

## Demos & Example Scripts

The `experiment_scripts/` directory contains ready-to-run scripts for common experiments, benchmarking, and ablation studies. These scripts demonstrate how to use the core pipeline for different scenarios, including:

- Multi-model scheduling and resource sharing
- Priority-based and round-robin inference
- Model complexity analysis
- LiDAR and image base experiments
- Multi-modal fusion experiments

To run a demo, simply execute the desired script, e.g.:
```bash
python3 experiment_scripts/bag_test.py
```

---

## Post-Processing & Analysis

The `tools/` directory provides a suite of tools for in-depth analysis of your experiments, including:

- **Layer-wise and kernel-level analysis** (e.g., `analyze_e2e_kernels.py`, `analyze_memcpy_kernels.py`)
- **Performance and sensitivity analysis** (e.g., `performance_analysis.py`, `dnn_sensitivity_analysis.py`)
- **Visualization and reporting** (e.g., `visualize_lidar_scene.py`, `priority_bar_plots.py`)
- **Scene and model variation analysis**

These tools help you interpret the results, identify bottlenecks, and optimize your models and pipelines.

---

## Key Tools

- **pPerf.py**  
  Profiling and performance monitoring utility for DNN inference.

- **inferencer.py**  
  Multi-model, multi-process inference node (LiDAR, image, or multi-modal).

- **inferencer_ms.py**  
  Multi-model, multi-threaded inference node.

- **sensor_publisher.py**  
  Publishes preloaded NuScenes data as ROS2 messages.

- **sensor_replayer.py (recommended)**  
  Replays recorded ROS2 bag files for offline benchmarking.

- **BEVInferencer.py**  
  Multi-modal (BEVFusion) inference utility.

---

## TODO

- [ ] Fix the multi-modal profiling pipeline

## Acknowledgments

perf_ws builds upon and integrates with several external tools and datasets:

- **LISA**: Atmospheric simulation and weather effects for autonomous driving scenarios (https://github.com/velatkilic/LISA)
- **nuscenes_to_rosbag**: Tools for converting NuScenes dataset to ROS2 bag format (https://github.com/WATonomous/nuscenes_to_ros2bag/blob/main/convert_to_ros2bag.sh)
- **NuScenes**: The 3D object detection dataset used for benchmarking and evaluation

We thank the respective authors and contributors for making these resources available to the community.


**Happy profiling! 🚗📊**
