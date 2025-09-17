## Installation

The easiest way to get started with perf_ws is using Docker. We provide a pre-configured Docker environment with all dependencies installed.

### Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t perf_ws -f Docker/pPerf.Dockerfile .
   ```

2. **Run the container:**
   ```bash
   docker run -it --gpus all -v $(pwd):/workspace perf_ws
   ```

3. **Inside the container, build the ROS2 workspace:**
   ```bash
   cd /workspace/perf_ws
   colcon build
   source install/setup.bash
   ```

### Manual Installation (Advanced)

If you prefer to install dependencies manually, you'll need:

- **Software:**
  - ROS2 (Humble or later)
  - Python 3.8+
  - PyTorch, MMDetection3D, OpenCV, NumPy, pandas, etc.
  - NVIDIA GPU (for CUDA profiling and inference)
  - NuScenes dataset (mini or full)

- **Important Library Versions:**
  - CUDA Driver: 525.60.13 or higher
  - CUDA Toolkit: 12.5
  - PyTorch: 2.0+ (with CUDA support)
  - MMDetection3D: Latest stable release
  - ROS2: Humble or later

- **NVIDIA GPU drivers and CUDA Toolkit:**
  - CUDA 12.5 requires 525.60.13 and higher.
  - Ensure that CUPTI is available on your path:
    ```bash
    $ /sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | grep libcupti
    ```
    If you don't see the correct `libcupti.so`, prepend its installation directory to your `LD_LIBRARY_PATH`:
    ```bash
    $ export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
    ```
    If this doesn't work, try:
    ```bash
    $ sudo apt-get install libcupti-dev
    ```
