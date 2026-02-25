#!/usr/bin/env python3
"""
Modular experiment runner for pPerf experiments.
"""

import os
import subprocess
import time
import pandas as pd
from subprocess import TimeoutExpired
from typing import Dict, List, Callable, Optional, Any
from pathlib import Path


class ExperimentRunner:
    """Handles running and post-processing of experiments."""
    
    def __init__(
        self,
        output_base: str,
        failure_log: str,
        nsys_base: List[str],
        timeout: int = 300,
        cleanup_memory: bool = True
    ):
        """
        Initialize experiment runner.
        
        Args:
            output_base: Base output directory
            failure_log: Path to failure log file
            nsys_base: Base nsys profiling command
            timeout: Timeout for experiment runs in seconds
            cleanup_memory: Whether to clean up memory after each experiment
        """
        self.output_base = output_base
        self.failure_log = failure_log
        self.nsys_base = nsys_base
        self.timeout = timeout
        self.cleanup_memory = cleanup_memory
    
    def run_experiments(
        self,
        df: pd.DataFrame,
        csv_path: str,
        config_updater: Callable[[pd.Series, int], str],
        launch_cmd_builder: Callable[[str], List[str]],
        skip_statuses: List[str] = ["run_success", "success"],
        max_runs: int = -1
    ) -> pd.DataFrame:
        """
        Run all pending experiments in the dataframe.
        
        Args:
            df: DataFrame with experiment configurations
            csv_path: Path to save updated CSV
            config_updater: Function that takes (row, index) and returns config_file path
            launch_cmd_builder: Function that takes config_file and returns ROS2 launch command
            skip_statuses: List of statuses to skip (already completed)
            max_runs: Maximum number of experiments to run. -1 means run all, >0 means run that many
            
        Returns:
            pd.DataFrame: Updated dataframe with execution results
        """
        print("\n" + "="*60)
        print("EXPERIMENT EXECUTION PHASE")
        if max_runs > 0:
            print(f"Running up to {max_runs} experiment(s)")
        else:
            print("Running all pending experiments")
        print("="*60)
        
        runs_completed = 0
        for i, row in df.iterrows():
            # Check if we've reached the maximum number of runs
            if max_runs > 0 and runs_completed >= max_runs:
                print(f"\nReached maximum number of runs ({max_runs}). Stopping execution.")
                break
            
            # Skip already completed experiments
            if df.at[i, "status"] in skip_statuses:
                print(f"Skipping run {i} - already completed (status: {df.at[i, 'status']})")
                continue
            
            # Clean up memory before starting experiment if enabled
            if self.cleanup_memory:
                from .memory_utils import clear_all_memory
                clear_all_memory()
            
            prefix = f"{self.output_base}/test_run_{i}"
            df.at[i, "status"] = "pending"
            
            # Update configuration
            try:
                config_file = config_updater(row, i)
                print(f"✓ Updated configuration for run {i}")
            except Exception as e:
                error_msg = f"Failed to update configuration for run {i}: {str(e)}"
                print(f"Error: {error_msg}")
                self._log_failure(error_msg)
                df.at[i, "status"] = "config_error"
                df.at[i, "start_time"] = time.time()
                df.to_csv(csv_path, index=False)
                continue
            
            # Build and run command
            ros2_cmd = launch_cmd_builder(config_file)
            full_cmd = self.nsys_base + ["-o", prefix] + ros2_cmd
            
            print(f"\n>>> Running Experiment ({i+1}/{len(df)}): {' '.join(full_cmd)}\n")
            
            start_time = time.time()
            try:
                subprocess.run(full_cmd, check=True, timeout=self.timeout)
                df.at[i, "status"] = "run_success"
                df.at[i, "start_time"] = start_time
                print(f"Experiment run {i} completed successfully")
            except TimeoutExpired as e:
                error_msg = f"Experiment run {i} timed out after {e.timeout} seconds"
                print(f"Error: {error_msg}")
                self._log_failure(error_msg)
                df.at[i, "status"] = "timeout"
                df.at[i, "start_time"] = start_time
            except Exception as e:
                error_msg = f"Experiment run {i} failed with unexpected error: {str(e)}"
                print(f"Error: {error_msg}")
                self._log_failure(error_msg)
                df.at[i, "status"] = "error"
                df.at[i, "start_time"] = start_time
            finally:
                df.to_csv(csv_path, index=False)
                print(f"Successfully saved status for run {i} to {csv_path}")
                runs_completed += 1
        
        print(f"\n✓ Completed {runs_completed} experiment run(s)")
        return df
    
    def post_process_experiments(
        self,
        df: pd.DataFrame,
        csv_path: str,
        nusc: Any,
        row_parser: Callable[[pd.Series], Dict[str, Any]],
        publish_mode: str = "bag",
        process_status: str = "run_success",
        cleanup_json: bool = True,
        process_kernels: bool = True,
        max_runs: int = -1
    ) -> pd.DataFrame:
        """
        Post-process all experiments that completed successfully.
        
        Args:
            df: DataFrame with experiment results
            csv_path: Path to save updated CSV
            nusc: NuScenes instance for timing analysis
            row_parser: Function that extracts scene and other info from a row
            publish_mode: Publishing mode for timing processor
            process_status: Status to look for to process experiments
            cleanup_json: Whether to delete JSON files after processing
            process_kernels: Whether to process and save individual kernel timings (default: True)
            max_runs: Maximum number of experiments to post-process. -1 means run all, >0 means run that many
            
        Returns:
            pd.DataFrame: Updated dataframe with post-processing results
        """
        from p_perf.post_process.timing_post import timing_processor
        
        print("\n" + "="*60)
        print("POST-PROCESSING PHASE")
        if max_runs > 0:
            print(f"Post-processing up to {max_runs} experiment(s)")
        else:
            print("Post-processing all completed experiments")
        print("="*60)
        
        processed_count = 0
        for i, row in df.iterrows():
            # Check if we've reached the maximum number of post-processing runs
            if max_runs > 0 and processed_count >= max_runs:
                print(f"\nReached maximum number of post-processing runs ({max_runs}). Stopping.")
                break
            
            # Only process experiments that completed the run but haven't been post-processed yet
            if df.at[i, "status"] != process_status:
                continue
            
            # Clean up memory before post-processing if enabled
            if self.cleanup_memory:
                from .memory_utils import clear_all_memory
                clear_all_memory()
            
            prefix = f"{self.output_base}/test_run_{i}"
            print(f"\n--- Processing Run {i}/{len(df)} ---")
            
            # Parse row to get necessary information
            row_data = row_parser(row)
            scene = row_data.get('scene')
            
            # Export nsys report to JSON if needed
            raw_timing_json = f"{prefix}.json"
            nsys_report = f"{prefix}.nsys-rep"
            
            if not os.path.exists(raw_timing_json):
                if os.path.exists(nsys_report):
                    print(f"Raw timing JSON file not found. Generating from {nsys_report}")
                    try:
                        subprocess.run([
                            "nsys", "export",
                            "--type", "json",
                            "--output", raw_timing_json,
                            nsys_report
                        ], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to export from {nsys_report}: {e}")
                        continue
                else:
                    print(f"Both {raw_timing_json} and {nsys_report} do not exist. Skipping.")
                    continue
            
            # Process timing data
            try:
                timing_analyzer = timing_processor(
                    nusc, raw_timing_json, self.output_base, i,
                    scene=scene, publish_mode=publish_mode, process_kernels=process_kernels
                )
                timing_analyzer.parse_json()
                print(f"timing analyzer finished parsing json")
                layer_records, kernel_records = timing_analyzer.generate_mapping()
                
                if process_kernels:
                    print(f"✓ Timing analysis completed for run {i} (layer + kernel timings)")
                else:
                    print(f"✓ Timing analysis completed for run {i} (layer timings only)")
            except Exception as e:
                print(f"✗ Timing analysis failed for run {i}: {e}")
                continue
            
            # Cleanup JSON file if requested
            if cleanup_json and os.path.exists(raw_timing_json):
                os.remove(raw_timing_json)
                print(f"Removed {raw_timing_json} to save disk space")
            
            # Cleanup timing analyzer
            timing_analyzer.cleanup()
            
            # Mark as fully complete
            df.at[i, "status"] = "success"
            df.to_csv(csv_path, index=False)
            
            print(f"✓ Run {i} post-processing completed and marked as success")
            processed_count += 1
            time.sleep(5)  # Brief pause between runs
        
        print(f"\n✓ Post-processed {processed_count} experiment(s)")
        return df
    
    def _log_failure(self, message: str):
        """Log failure message to failure log file."""
        with open(self.failure_log, "a") as flog:
            flog.write(f"{message}\n")
    
    @staticmethod
    def build_ros2_launch_cmd(
        package: str,
        launch_file: str,
        config_file: str,
        **kwargs
    ) -> List[str]:
        """
        Build a ROS2 launch command.
        
        Args:
            package: ROS2 package name
            launch_file: Launch file name
            config_file: Path to config file
            **kwargs: Additional launch arguments
            
        Returns:
            List[str]: Command list
        """
        cmd = ["ros2", "launch", package, launch_file, f"config_file:={config_file}"]
        for key, value in kwargs.items():
            cmd.append(f"{key}:={value}")
        return cmd
    
    @staticmethod
    def create_nsys_base_cmd(
        traces: List[str] = ["cuda", "nvtx", "cudnn"],
        backtrace: str = "none",
        force_overwrite: bool = True
    ) -> List[str]:
        """
        Create base nsys profiling command.
        
        Args:
            traces: List of trace types
            backtrace: Backtrace mode
            force_overwrite: Whether to force overwrite existing files
            
        Returns:
            List[str]: Base nsys command
        """
        return [
            "nsys", "profile",
            f"--trace={','.join(traces)}",
            f"--backtrace={backtrace}",
            "--force-overwrite", str(force_overwrite).lower(),
        ]

