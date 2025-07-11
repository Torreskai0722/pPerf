#!/bin/bash

# run_all.sh - Run both bag_test.py and ms_test.py sequentially with system cleanup
# Author: System
# Date: $(date)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to clean up system resources
cleanup_system() {
    log "Starting system cleanup..."
    
    # Store current script PIDs to avoid killing them
    local current_script_pid=""
    if [ -n "$1" ]; then
        current_script_pid="$1"
    fi
    
    # Function to kill processes excluding current PID
    kill_processes_excluding() {
        local pattern="$1"
        local exclude_pid="$2"
        
        if [ -n "$exclude_pid" ]; then
            # Get PIDs matching pattern, exclude current script PID
            pgrep -f "$pattern" | while read pid; do
                if [ "$pid" != "$exclude_pid" ]; then
                    kill "$pid" 2>/dev/null || true
                fi
            done
        else
            # Kill all matching processes
            pkill -f "$pattern" || true
        fi
    }
    
    # Kill any remaining ROS2 processes (but not the current one)
    log "Killing ROS2 processes..."
    kill_processes_excluding "ros2 launch p_perf" "$current_script_pid"
    kill_processes_excluding "ros2" "$current_script_pid"
    
    # Kill any remaining Python processes related to our experiments (but not the current one)
    log "Killing Python experiment processes..."
    kill_processes_excluding "bag_test.py" "$current_script_pid"
    kill_processes_excluding "ms_test.py" "$current_script_pid"
    
    # Clear GPU memory if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        log "Clearing GPU memory..."
        nvidia-smi --gpu-reset || true
    fi
    
    # Clear system cache
    log "Clearing system cache..."
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    # Wait for processes to fully terminate
    log "Waiting for processes to terminate..."
    sleep 5
    
    # Function to force kill processes excluding current PID
    force_kill_processes_excluding() {
        local pattern="$1"
        local exclude_pid="$2"
        
        if [ -n "$exclude_pid" ]; then
            # Get PIDs matching pattern, exclude current script PID
            pgrep -f "$pattern" | while read pid; do
                if [ "$pid" != "$exclude_pid" ]; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done
        else
            # Force kill all matching processes
            pkill -9 -f "$pattern" || true
        fi
    }
    
    # Check if any processes are still running (but not the current one)
    if [ -n "$current_script_pid" ]; then
        if pgrep -f "ros2 launch p_perf" | grep -v "$current_script_pid" > /dev/null; then
            warning "Some ROS2 processes are still running, force killing..."
            force_kill_processes_excluding "ros2 launch p_perf" "$current_script_pid"
            sleep 2
        fi
        
        if pgrep -f "bag_test.py\|ms_test.py" | grep -v "$current_script_pid" > /dev/null; then
            warning "Some Python processes are still running, force killing..."
            force_kill_processes_excluding "bag_test.py\|ms_test.py" "$current_script_pid"
            sleep 2
        fi
    else
        if pgrep -f "ros2 launch p_perf" > /dev/null; then
            warning "Some ROS2 processes are still running, force killing..."
            pkill -9 -f "ros2 launch p_perf" || true
            sleep 2
        fi
        
        if pgrep -f "bag_test.py\|ms_test.py" > /dev/null; then
            warning "Some Python processes are still running, force killing..."
            pkill -9 -f "bag_test.py\|ms_test.py" || true
            sleep 2
        fi
    fi
    
    success "System cleanup completed"
}

# Function to check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Check available disk space
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10000000 ]; then  # Less than 10GB
        error "Low disk space available: ${available_space}KB"
        return 1
    fi
    
    # Check available memory
    available_mem=$(free -k | awk 'NR==2 {print $7}')
    if [ "$available_mem" -lt 1000000 ]; then  # Less than 4GB
        error "Low memory available: ${available_mem}KB"
        return 1
    fi
    
    success "System resources check passed"
}

# Function to run a Python script with error handling
run_python_script() {
    local script_name=$1
    local script_path=$2
    
    log "Starting $script_name..."
    
    # Check if script exists
    if [ ! -f "$script_path" ]; then
        error "Script $script_path not found!"
        return 1
    fi
    
    # Run the script and capture its PID
    python3 "$script_path" &
    local script_pid=$!
    
    # Wait for the script to complete
    if wait $script_pid; then
        success "$script_name completed successfully"
        return 0
    else
        error "$script_name failed with exit code $?"
        return 1
    fi
}

# Main execution
main() {
    log "Starting run_all.sh - Sequential execution of bag_test.py and ms_test.py"
    log "Current directory: $(pwd)"
    
    # Get the current script PID to avoid self-killing
    local current_pid=$$
    log "Current script PID: $current_pid"
    
    # Check if we're in the right directory
    if [ ! -f "bag_test.py" ] || [ ! -f "ms_test.py" ]; then
        error "bag_test.py or ms_test.py not found in current directory!"
        error "Please run this script from the experiment_scripts directory"
        exit 1
    fi
    
    # Initial system check
    check_system_resources || {
        error "System resource check failed. Aborting."
        exit 1
    }
    
    # Initial cleanup (pass current PID to avoid self-killing)
    cleanup_system "$current_pid"
    
    # Run bag_test.py
    log "=== PHASE 1: Running bag_test.py ==="
    if run_python_script "bag_test.py" "./bag_test.py"; then
        success "Phase 1 completed successfully"
    else
        error "Phase 1 failed. Continuing with cleanup and Phase 2..."
    fi
    
    # Cleanup between phases (pass current PID to avoid self-killing)
    log "=== CLEANUP BETWEEN PHASES ==="
    cleanup_system "$current_pid"
    
    # Wait a bit more for complete cleanup
    log "Additional wait time for complete cleanup..."
    sleep 10
    
    # Check system resources again
    check_system_resources || {
        warning "System resource check failed after Phase 1, but continuing..."
    }
    
    # Run ms_test.py
    log "=== PHASE 2: Running ms_test.py ==="
    if run_python_script "ms_test.py" "./ms_test.py"; then
        success "Phase 2 completed successfully"
    else
        error "Phase 2 failed."
    fi
    
    # Final cleanup (pass current PID to avoid self-killing)
    log "=== FINAL CLEANUP ==="
    cleanup_system "$current_pid"
    
    # Summary
    log "=== EXECUTION SUMMARY ==="
    success "run_all.sh completed!"
    log "Both scripts have been executed with proper cleanup between phases"
    log "Check the output directories for results:"
    log "  - Bag test results: /mmdetection3d_ros2/outputs/Image_Lidar_full"
    log "  - MS test results: /mmdetection3d_ros2/outputs/ms"
}

# Handle script interruption
trap 'echo -e "\n${RED}[INTERRUPTED]${NC} Script interrupted by user"; cleanup_system $$; exit 1' INT TERM

# Run main function
main "$@"
