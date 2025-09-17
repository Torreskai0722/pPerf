#!/bin/bash
printf "Disabling MPS (Multi-Process Service)\n"
printf "=====================================\n"

# Kill any running MPS daemons
printf "Shutting down MPS daemon...\n"
echo quit | nvidia-cuda-mps-control 2>/dev/null || printf "No MPS daemon was running\n"
sleep 1

# Force kill any remaining MPS processes
printf "Cleaning up MPS processes...\n"
sudo pkill -9 -f nvidia-cuda-mps
sudo rm -rf /tmp/nvidia-mps /tmp/nvidia-log

# Restore GPU to default mode
printf "Setting GPU to default mode...\n"
nvidia-smi -i 0 -c DEFAULT

printf "✓ MPS is now DISABLED\n"
printf "  - GPU restored to default mode\n"
printf "  - All MPS processes terminated\n"
printf "  - MPS directories cleaned up\n"
printf "\n"
printf "GPU is now in standard mode for regular CUDA applications.\n" 