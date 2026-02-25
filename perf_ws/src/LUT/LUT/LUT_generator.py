import sys
import os
import subprocess
import pandas as pd

# Update this path to match your Nsight Compute 2025 installation
NCU_PYTHON_PATH = "/usr/local/NVIDIA-Nsight-Compute-2025.1/extras/python/"
sys.path.append(NCU_PYTHON_PATH)

try:
    import ncu_report
except ImportError:
    print(f"Error: Could not find ncu_report API at {NCU_PYTHON_PATH}")
    sys.exit(1)

def extract_nvtx_info(action):
    # 1. Get the NVTX state from the action (kernel)
    nvtx_state = action.nvtx_state()
    
    # 2. Get domains (default domain is usually 0)
    domains = nvtx_state.domains()
    
    range_names = []
    for domain_id in domains:
        domain_info = nvtx_state.domain_by_id(domain_id)
        
        # 3. Get the stack of Push/Pop ranges
        # This returns a tuple of range names from bottom to top
        push_pop = domain_info.push_pop_ranges()
        if push_pop:
            range_names.extend(push_pop)
            
    # Return as a single string (e.g., "Model/Layer/Conv")
    return "/".join(range_names) if range_names else "N/A"
    

def run_ncu_profiling(output_report, script_path):
    """
    Run Nsight Compute profiling on the ops_replayer script.
    
    Args:
        output_report: Output report filename (without extension)
        script_path: Path to the Python script to profile
        
    Returns:
        Path to the generated report file
    """
    sections = ["LaunchStats", "Occupancy"]
    nvtx_filter = "regex:.* \| engine_config_.*/"
    # Build the NCU command
    cmd = [
        "ncu",
        "-o", output_report,
        "--force-overwrite",
        "--nvtx",
        "--nvtx-include", nvtx_filter,
    ]
    
    # # Add sections
    for section in sections:
        cmd.extend(["--section", section])
    
    # Add the Python command
    cmd.extend(["python3", script_path])
    
    print("=" * 80)
    print("Running Nsight Compute Profiling...")
    print("Command:", " ".join(cmd))
    print("=" * 80)
    
    try:
        # Run the profiling command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        
        report_path = f"{output_report}.ncu-rep"
        print(f"\n{'=' * 80}")
        print(f"Profiling completed successfully!")
        print(f"Report saved to: {report_path}")
        print("=" * 80)
        
        return report_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error running NCU profiling: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'ncu' command not found. Make sure Nsight Compute is installed and in PATH.")
        sys.exit(1)


def extract_conv_metrics(report_path, output_csv):
    """
    Extract convolution kernel metrics from NCU report.
    
    Args:
        report_path: Path to the .ncu-rep file
        output_csv: Path to save the CSV output
    """
    if not os.path.exists(report_path):
        print(f"Error: Report file {report_path} not found.")
        return

    print(f"\n{'=' * 80}")
    print(f"Analyzing report: {report_path}")
    print("=" * 80)

    # Load the report
    report = ncu_report.load_report(report_path)
    extracted_data = []
    
    print(f"Total NVTX ranges found: {report.num_ranges()}")

    # Iterate through all actions in the report (not ranges)
    # Since we're extracting NVTX info from each action directly
    total_actions = 0
    for range_idx in range(report.num_ranges()):
        current_range = report.range_by_idx(range_idx)
        total_actions += current_range.num_actions()
    
    print(f"Total actions to process: {total_actions}")
    
    action_count = 0
    for range_idx in range(report.num_ranges()):
        current_range = report.range_by_idx(range_idx)
        
        # Iterate through kernels in this range
        for action_idx in range(current_range.num_actions()):
            action = current_range.action_by_idx(action_idx)
            kernel_name = action.name()
            action_count += 1

            # Filter for convolutional kernels
            if kernel_name in ["nhwcToNchwKernel", "nchwToNhwcKernel"]:
                continue
                
            try:
                # Extract NVTX info using the new function
                nvtx_range_name = extract_nvtx_info(action)
                
                # Parse the NVTX range name
                # Format: "op_id_scale1280x720_input640x352 | engine_config_0"
                engine_config = None
                op_id = None
                scale_info = None
                input_shape = None
                
                if " | engine_config_" in nvtx_range_name:
                    # Pattern: "op_id_scale1280x720_input640x352 | engine_config_XXX"
                    parts = nvtx_range_name.split(" | ")
                    if len(parts) >= 2:
                        full_op_id = parts[0].strip()
                        engine_config = parts[1].strip()
                        
                        # Parse the full_op_id to extract op_id, scale, and input shape
                        if "_scale" in full_op_id and "_input" in full_op_id:
                            op_and_scale = full_op_id.split("_scale")
                            op_id = op_and_scale[0]
                            
                            scale_and_input = op_and_scale[1].split("_input")
                            scale_info = scale_and_input[0]
                            input_shape = scale_and_input[1] if len(scale_and_input) > 1 else None
                        elif "_input" in full_op_id:
                            op_and_input = full_op_id.split("_input")
                            op_id = op_and_input[0]
                            input_shape = op_and_input[1] if len(op_and_input) > 1 else None
                        else:
                            op_id = full_op_id
                
                # Helper function to safely get metric value
                def get_metric(action, metric_name):
                    try:
                        if metric_name in action:
                            return action[metric_name].value()
                    except:
                        pass
                    return None
                
                metrics = {
                    "NVTX Range": nvtx_range_name,
                    "Op ID": op_id if op_id else "N/A",
                    "Scale": scale_info if scale_info else "N/A",
                    "Input Shape": input_shape if input_shape else "N/A",
                    "Engine Config": engine_config if engine_config else "N/A",
                    "Kernel Name": kernel_name,
                    "Duration (ns)": get_metric(action, "gpu__time_duration.sum"),
                    "Registers Per Thread": get_metric(action, "launch__registers_per_thread"),
                    "Static Shared Memory (B)": get_metric(action, "launch__shared_mem_per_block_static"),
                    "Dynamic Shared Memory (B)": get_metric(action, "launch__shared_mem_per_block_dynamic"),
                    "Block Size": get_metric(action, "launch__block_size"),
                    "Grid Size": get_metric(action, "launch__grid_size"),
                }
                extracted_data.append(metrics)
                
                if action_count % 100 == 0:
                    print(f"  Processed {action_count}/{total_actions} actions...")
                
            except Exception as e:
                print(f"  ✗ Warning: Could not extract metrics for kernel {kernel_name}: {e}")
                continue

    if not extracted_data:
        print("Warning: No convolutional kernels found in the report.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(extracted_data)
    
    # Sort by Op ID, Scale, and Engine Config for better readability
    sort_cols = []
    if "Op ID" in df.columns:
        sort_cols.append("Op ID")
    if "Scale" in df.columns:
        sort_cols.append("Scale")
    if "Engine Config" in df.columns:
        sort_cols.append("Engine Config")
    
    if sort_cols:
        df = df.sort_values(by=sort_cols)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"Analysis Complete!")
    print(f"Extracted {len(df)} convolutional kernel entries")
    print(f"Results saved to: {output_csv}")
    print("=" * 80)
    
    # Print summary statistics
    print(f"\nSummary:")
    if "Op ID" in df.columns:
        unique_ops = df["Op ID"].nunique()
        print(f"  - Unique operations: {unique_ops}")
    if "Scale" in df.columns:
        unique_scales = df["Scale"].nunique()
        print(f"  - Unique scales: {unique_scales}")
        if unique_scales > 0:
            scales_list = df["Scale"].unique()
            print(f"  - Scales: {', '.join(str(s) for s in scales_list if s != 'N/A')}")
    if "Engine Config" in df.columns:
        unique_engines = df["Engine Config"].nunique()
        print(f"  - Unique engine configurations: {unique_engines}")


if __name__ == "__main__":
    # Configuration
    report_name = "my_profile_report"
    output_csv = "conv_kernel_analysis.csv"
    script_path = "src/LUT/LUT/ops_replayer.py"
    
    print("=" * 80)
    print("LUT Generator - Automated Profiling and Analysis")
    print("=" * 80)
    
    # Step 1: Run NCU profiling
    report_path = run_ncu_profiling(report_name, script_path)
    
    # Step 2: Analyze the report
    extract_conv_metrics(report_path, output_csv)
    
    print("\n" + "=" * 80)
    print("LUT Generation Complete!")
    print("=" * 80)