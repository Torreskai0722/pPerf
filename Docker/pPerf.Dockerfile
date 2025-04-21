# Use Ubuntu 22.04 with CUDA 11.8 and cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        dbus \
        fontconfig \
        gnupg2 \
        libasound2 \
        libfreetype6 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libxcb-cursor0 \
        xcb \
        xkb-data \
        openssh-client \
        wget \
        vim \
        ffmpeg \
        git \
        ninja-build \
        curl \
        lsb-release \
        python3-pip \
        python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U colcon-common-extensions


# Install specific version of empy (3.3.4) to avoid compatibility issues
RUN pip3 install --no-cache-dir --force-reinstall empy==3.3.4

# Install numpy PyTorch with CUDA 11.8 
RUN pip install --no-cache-dir "numpy<2" --force-reinstall
RUN pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MMEngine, MMCV, MMDetection
RUN pip install openmim && \
    mim install "mmengine"

RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

RUN mim install "mmdet>=3.0.0"

# Install MMDetection3D
RUN git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x /mmdetection3d \
    && cd /mmdetection3d \
    && pip install --no-cache-dir -e .

# Install spconv so that will not have issues with BEVFusion
RUN pip install spconv-cu118

# ----------------------
# Install ROS 2 Humble
# ----------------------
# Set up ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu `lsb_release -cs` main" | tee /etc/apt/sources.list.d/ros2.list

# Set timezone to avoid interactive prompt during installation
ENV DEBIAN_FRONTEND=noninteractive TZ=America/New_York

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rqt* \
    python3-rosdep \
    python3-colcon-common-extensions \
    ros-humble-rosbag2-storage-mcap \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pynvml

# Initialize rosdep
RUN rosdep init && rosdep update

# Set up ROS 2 environment
ENV ROS_DISTRO=humble
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV PATH=${ROS_ROOT}/bin:$PATH
ENV LD_LIBRARY_PATH=${ROS_ROOT}/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=${ROS_ROOT}/lib/python3.10/site-packages:$PYTHONPATH
ENV CMAKE_PREFIX_PATH=${ROS_ROOT}:$CMAKE_PREFIX_PATH

# Source ROS 2 environment on container start
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc


# ----------------------
# Install Nsys
# ----------------------
RUN cd /tmp && \
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/nsight-systems-2025.2.1_2025.2.1.130-1_amd64.deb && \
    apt-get install -y ./nsight-systems-2025.2.1_2025.2.1.130-1_amd64.deb && \
    rm -rf /tmp/*
# Set working directory
WORKDIR /mmdetection3d_ros2
