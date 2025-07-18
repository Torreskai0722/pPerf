# pPerf: Multi-Tenant DNN Inference Predictability Profiler

pPerf is a suite of tools for profiling, benchmarking, and analyzing multi-model DNN inference pipelines for autonomous driving. The toolkit supports single- and multi-model inference with LiDAR, image, and multi-modal (BEVFusion) models using ROS 2 for real-time and offline replay. It includes profiling tools for timing, GPU/CPU usage, and NVTX-based CUDA tracing. Data can be published from NuScenes or replayed from ROS 2 bags. Logs include timing, predictions, and resource usage in JSON/CSV. Post-processing tools provide kernel, layer, and system-level analysis, with scripts for running standard benchmarks.

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

pPerf builds upon and integrates atop: [LISA](https://github.com/velatkilic/LISA), [nuscenes_to_rosbag](https://github.com/WATonomous/nuscenes_to_ros2bag), [NuScenes](https://www.nuscenes.org/), [DINO](https://github.com/IDEA-Research/DINO?tab=readme-ov-file).


**Happy profiling! 🚗📊**
