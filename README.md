# pPerf: Multi-Tenant DNN Inference Predictability Profiler

pPerf is a suite of tools for profiling, benchmarking, and analyzing multi-model DNN inference pipelines for autonomous driving. It supports both LiDAR and camera (image) modalities, and is designed for use with the NuScenes dataset. perf_ws helps you understand, debug, and optimize inference pipelines running on CPUs and GPUs.

## Features

perf_ws offers a number of tools to analyze and visualize the performance of your models across multiple GPU and data streams. Some of the key features include:

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

## Quick Start

1. **Install**  
   Follow this to setup: [INSTALL.md](doc/INSTALL.md).

2. **Prepare your data**  
   Download and extract the NuScenes dataset. Update paths in your config if needed. See [DATA.md](DATA.md).

3. **Launch the pipeline**

   - **Start the data publisher:**
     ```bash
     ros2 run p_perf sensor_publisher.py
     ```

   - **Start the inference node (single or multi-model):**
     ```bash
     ros2 run p_perf inferencer.py
     # or
     ros2 run p_perf inferencer_ms.py
     ```

   - **(Optional) Replay a bag file:**
     ```bash
     ros2 run p_perf sensor_replayer.py
     ```

4. **View results**  
   - Profiling outputs (timing, GPU/CPU stats) are saved in your specified data directory.
   - Analyze logs and outputs for performance, accuracy, and resource usage.

## Demos

The `experiment_scripts/` directory contains ready-to-run scripts for common experiments, benchmarking, and ablation studies. These scripts demonstrate how to use the core pipeline for different scenarios, including:

- Multi-model scheduling and resource sharing
- Priority-based and round-robin inference
- Model complexity analysis
- LiDAR and image base experiments
- Multi-modal fusion experiments

For detailed documentation on the experiment scripts, see [experiment_scripts/README.md](experiment_scripts/README.md).

To run a demo, simply execute the desired script, e.g.:
```bash
python experiment_scripts/bag_test.py
```

## Post-Processing & Analysis

The `tools/` directory provides a suite of tools for in-depth analysis of your experiments, including:

- **Layer-wise and kernel-level analysis** (e.g., `analyze_e2e_kernels.py`, `analyze_memcpy_kernels.py`)
- **Performance and sensitivity analysis** (e.g., `performance_analysis.py`, `dnn_sensitivity_analysis.py`)
- **Visualization and reporting** (e.g., `visualize_lidar_scene.py`, `priority_bar_plots.py`)
- **Scene and model variation analysis**

These tools help you interpret the results, identify bottlenecks, and optimize your models and pipelines.

## TODO

- [ ] Fix the multi-modal profiling pipeline

## Acknowledgments

pPerf builds upon and integrates atop: **LISA**, **nuscenes_to_rosbag**, **NuScenes**, **DINO**.

**Happy profiling! 🚗📊**
