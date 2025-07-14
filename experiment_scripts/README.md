# Experiment Scripts

This directory contains ready-to-run scripts for common experiments, benchmarking, and ablation studies in perf_ws.

## Overview

The experiment scripts demonstrate how to use the core perf_ws pipeline for different scenarios and research questions. Each script is designed to be run independently and produces comprehensive results for analysis.

## Available Scripts

### Core Experiment Scripts

- **`bag_test.py`** - Basic bag replay experiment with multi-model, multi-process inference
- **`ms_test.py`** - Multi-model, multi-threaded inference testing
- **`lidar_base.py`** - LiDAR-only model benchmarking as best scenario case
- **`image_base.py`** - Image-only model benchmarking as best scenario case
- **`multi_modal.py`** - Multi-modal (LiDAR + camera) fusion experiments

### Scheduling and Resource Management

- **`bag_test_priority.py`** - Priority-based scheduling experiments
- **`bag_test_RR.py`** - Round-robin scheduling experiments
- **`setup_scheduling.py`** - Configure and setup different scheduling policies

### Analysis and Utilities

- **`model_complexity.py`** - Model complexity analysis and comparison
- **`cleanup_utils.py`** - Utilities for cleaning up experiment outputs
- **`run_all.sh`** - Shell script to run all experiments sequentially

## Usage

### Basic Usage

Run any experiment script directly:

```bash
python experiment_scripts/bag_test.py
```

### Running Multiple Experiments

Use the provided shell script to run all experiments:

```bash
bash experiment_scripts/run_all.sh
```

### Customizing Experiments

Each script accepts command-line arguments for customization:

```bash
python experiment_scripts/bag_test.py --scene <scene_token> --model <model_name>
```

## Experiment Types

### 1. Single Model Experiments

**Purpose**: Benchmark individual models in isolation
**Scripts**: `lidar_base.py`, `image_base.py`
**Outputs**: Timing logs, detection results, resource usage

### 2. Multi-Model Experiments

**Purpose**: Test concurrent execution of multiple models
**Scripts**: `ms_test.py`, `bag_test_priority.py`, `bag_test_RR.py`
**Outputs**: Comparative timing, resource sharing analysis, scheduling efficiency

### 3. Multi-Modal Experiments

**Purpose**: Evaluate fusion of LiDAR and camera data
**Scripts**: `multi_modal.py`
**Outputs**: Fusion performance, synchronization analysis

### 4. Complexity Analysis

**Purpose**: Understand model computational requirements
**Scripts**: `model_complexity.py`
**Outputs**: FLOPs, MACs, layer-wise analysis

## Configuration

### Environment Variables

Set these environment variables before running experiments:

```bash
export DATA_ROOT=/path/to/nuscenes/dataset
export OUTPUT_DIR=/path/to/output/directory
export BAG_DIR=/path/to/bag/files
```

### Model Configuration

Update model configurations in the scripts:

```python
# Example model configuration
LIDAR_MODELS = [
    'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d',
    'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d'
]

IMAGE_MODELS = [
    'faster-rcnn_r50_fpn_1x_coco',
    'yolox_s_8x8_300e_coco'
]
```

## Output Structure

Each experiment generates:

```
outputs/
  experiment_name/
    lidar_pred_*.json      # LiDAR detection results
    image_pred_*.json      # Image detection results
    delays_*.csv          # Timing measurements
    gpu_*.csv            # GPU utilization logs
    cpu_*.csv            # CPU utilization logs
    config_*.json        # Experiment configuration
```

## Analysis

After running experiments, use the post-processing tools at tools/ dir:

```bash
cd post_processing/
python performance_analysis.py --input_dir ../outputs/experiment_name/
python analyze_e2e_kernels.py --input_dir ../outputs/experiment_name/
```