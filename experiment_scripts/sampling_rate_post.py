import subprocess
import csv
import os
from sampling_rate_graphing import filter_indices, read_csv
# Path to the CSV file
INPUT_DIR = "sampling_log_1"
csv_path = f"/mmdetection3d_ros2/pPerf_ws/{INPUT_DIR}/param_mapping.csv"

rows = read_csv(csv_path)
selected_indices = filter_indices(rows)

# Run the command for each selected index
for i in range(123, 144):
    run_path = f"/mmdetection3d_ros2/pPerf_ws/{INPUT_DIR}/v1_3_run_{i}"
    command = ["python3", "src/pPerf_v1_3/pPerf_v1_3/pPerf_post.py", run_path]

    print(f"\nRunning: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Delete the corresponding .json file after processing
    json_path = f"{run_path}.json"
    try:
        os.remove(json_path)
        print(f"Deleted: {json_path}")
    except Exception as e:
        print(f"Error deleting {json_path}: {e}")
