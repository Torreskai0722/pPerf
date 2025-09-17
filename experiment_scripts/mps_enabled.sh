#!/bin/bash
printf "Enabling MPS (Multi-Process Service)\n"
printf "====================================\n"

# Kill any stale daemons
printf "Cleaning up any existing MPS processes...\n"
sudo pkill -9 -f nvidia-cuda-mps
sudo rm -rf /tmp/nvidia-mps /tmp/nvidia-log

# Put GPU in exclusive mode
printf "Setting GPU to exclusive process mode...\n"
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# Export MPS env (server + clients must see same directories)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Start MPS daemon
printf "Starting MPS daemon...\n"
nvidia-cuda-mps-control -d
sleep 2   # give server time to initialize

printf "✓ MPS is now ENABLED and ready for use\n"
printf "  - GPU in exclusive process mode\n"
printf "  - MPS daemon running\n"
printf "  - Pipe directory: /tmp/nvidia-mps\n"
printf "  - Log directory: /tmp/nvidia-log\n"
printf "\n"
printf "You can now run your experiments with MPS support.\n"
printf "Use 'experiments_mps_disabled.sh' to disable MPS when done.\n" 