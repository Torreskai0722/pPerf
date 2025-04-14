import json
import pandas as pd
import sys

COPYKIND_MAPPING = {1: "host2device", 8: "device2device", 2: "device2host"}

def decode_globalid(global_id):
    """
    Decode pid from given decoded globalTid/globalPid 
    encoded 0xHHVVPPPPPPTTTTTT
        H: hardware device code
        V: virtual machine code
        P: process ID
        T: thread ID

    :param global_id: can either be globalTid or globalPid
    :return: return PID and TID of the given id, however, TID don't mean anything if it is a globalPid 
    as the globalPid's value has no meaning, always 0
    """
    PID = (global_id >> 24) & 0xFFFFFF  # Extract PID (24 bits)
    TID = global_id & 0xFFFFFF  # Extract TID (24 bits)
    return PID, TID

def process_nvtx_event(entry, nvtx_events, pid_map):
    """Extracts and stores NVTX event information."""
    nvtx = entry["NvtxEvent"]
    
    parts = nvtx["Text"].split('.')
    if parts[-1] == 'e2e':
        input_name, model_name, *layer = parts
    else:
        model_name, *layer = parts
        input_name = "pending"  # Mark as pending for later post-processing

    pid, _ = decode_globalid(int(nvtx["GlobalTid"]))
    start, end = int(nvtx["Timestamp"]), int(nvtx["EndTimestamp"])
    
    nvtx_events.append({
        "Model Name": model_name,
        "Input": input_name,
        "Layer": '.'.join(layer),
        "StartTimestamp": start,
        "EndTimestamp": end,
        "Elapsed": (end - start) * 1e-6,
        "PID": pid
    })
    pid_map.setdefault(pid, set()).add(model_name)

def fill_pending_inputs(nvtx_events):
    """
    For each NVTX event with input 'pending', fill it based on enclosing input events
    with the same PID and model name.
    """
    input_events = [
        e for e in nvtx_events 
        if e["Input"] != "pending"
    ]

    for event in nvtx_events:
        if event["Input"] != "pending":
            continue
        
        pid, model, start, end = event["PID"], event["Model Name"], event["StartTimestamp"], event["EndTimestamp"]
        
        # Find matching input events with same PID & model name, whose timestamp range encloses the current event
        candidates = [
            e for e in input_events
            if e["PID"] == pid and e["Model Name"] == model and e["StartTimestamp"] <= start <= e["EndTimestamp"]
            and e["StartTimestamp"] <= end <= e["EndTimestamp"]
        ]
        
        if candidates:
            # Pick the earliest enclosing input
            matched_input = sorted(candidates, key=lambda x: x["StartTimestamp"])
            if len(matched_input) > 1:
                print(candidates, '\n')
            event["Input"] = matched_input[0]["Input"]

def process_cuda_event(entry, cuda_events, kernel_name_list):
    """
    Extracts and stores CUDA event information for kernel operations (eventClass 3)
    and memory copy operations (eventClass 1).
    """
    cuda = entry["CudaEvent"]
    eventClass = cuda.get("eventClass")
    pid, _ = decode_globalid(int(cuda["globalPid"]))

    memcpy_size = 0

    if eventClass == 3:  # Kernel operation
        kernel_info = cuda.get("kernel")
        kernel_index = int(kernel_info.get("shortName"))
        kernel_name = kernel_name_list[kernel_index]

    elif eventClass == 1:  # Memory operation
        mem_cpy = cuda.get("memcpy")
        kernel_name = COPYKIND_MAPPING.get(int(mem_cpy.get("copyKind")))
        memcpy_size = int(mem_cpy.get("sizebytes", 0))
    
    else:
        return

    cuda_events.append({
        "Kernel Name": kernel_name, 
        "Kernel Start": int(cuda.get("startNs", 0)),
        "Kernel End": int(cuda.get("endNs", 0)),
        "Kernel Elapsed": (int(cuda.get("endNs", 0)) - int(cuda.get("startNs", 0))) * 1e-6,
        "Memcpy Size": memcpy_size, 
        "CorrelationId": cuda.get("correlationId"),
        "PID": pid
    })


def process_trace_event(entry, trace_process_events):
    """Extracts and stores trace process event information."""
    trace = entry["TraceProcessEvent"]
    pid, _ = decode_globalid(int(trace["globalTid"]))
    trace_process_events[trace["correlationId"]] = (int(trace["startNs"]), pid)

def process_json(json_file_path):
    """Reads and processes the JSON file line by line."""
    nvtx_events = []
    cuda_events = []
    pid_map = {}
    kernel_name_list = []
    trace_process_events = {}
    
    with open(json_file_path, "r") as file:
        for i, line in enumerate(file):
            try:
                entry = json.loads(line.strip())
                if i == 0 and "data" in entry:
                    kernel_name_list = entry["data"]
                    continue
                
                if "NvtxEvent" in entry:
                    process_nvtx_event(entry, nvtx_events, pid_map)
                elif "CudaEvent" in entry:
                    process_cuda_event(entry, cuda_events, kernel_name_list)
                elif "TraceProcessEvent" in entry:
                    process_trace_event(entry, trace_process_events)
            except json.JSONDecodeError:
                continue
    
    return nvtx_events, cuda_events, pid_map, trace_process_events

def compute_gpu_time(candidate_kernels):
    """Compute GPU active time using a union-of-intervals approach."""
    if candidate_kernels.empty:
        return 0

    # Extract start and end times
    intervals = sorted(zip(candidate_kernels["Kernel Start"], candidate_kernels["Kernel End"]))

    # Merge overlapping intervals
    total_time = 0
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start > current_end:  # No overlap, add previous interval time
            total_time += (current_end - current_start)
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)  # Merge overlapping intervals

    # Add last merged interval
    total_time += (current_end - current_start)
    
    return total_time * 1e-6  # Convert to milliseconds


def generate_mapping(nvtx_df, cuda_df, trace_process_events):
    mapping = []
    
    for _, nvtx in nvtx_df.iterrows():
        pid, start, end = nvtx["PID"], nvtx["StartTimestamp"], nvtx["EndTimestamp"]
        
        trace_candidates = {
            cid: ts for cid, (ts, p) in trace_process_events.items() 
            if start <= ts <= end and p == pid
        }

        candidate_kernels = cuda_df[
            (cuda_df["CorrelationId"].isin(trace_candidates.keys())) & 
            (cuda_df["PID"] == pid)
        ]

        kernel_inference = candidate_kernels.groupby("Kernel Name")["Kernel Elapsed"].sum().to_dict()
        memcpy_kernels = candidate_kernels[candidate_kernels["Kernel Name"].isin(COPYKIND_MAPPING.values())]
        memcpy_bytes = memcpy_kernels.groupby("Kernel Name")["Memcpy Size"].sum().to_dict()
        
        # Fix GPU turnaround time: First start to last end
        layer_gpu_turnaround = (candidate_kernels["Kernel End"].max() - candidate_kernels["Kernel Start"].min()) * 1e-6 if not candidate_kernels.empty else 0

        # Fix GPU computation time: Sum of non-overlapping kernel intervals
        gpu_active_time = compute_gpu_time(candidate_kernels)

        # GPU Wait Time: Turnaround time - actual execution time
        layer_gpu_waittime = layer_gpu_turnaround - gpu_active_time
        layer_cpu_time = (end - start) * 1e-6
        layer_memcpy_bytes = sum(memcpy_bytes.values())

        mapping.append({
            "Input Data Name": nvtx["Input"],
            "Model Name": nvtx["Model Name"],
            "Layer Name": nvtx["Layer"],
            "Layer Elapsed Time": layer_cpu_time,
            "Layer GPU Turnaround Time": layer_gpu_turnaround,
            "Layer GPU Computation Time": gpu_active_time,
            "Layer GPU Wait Time": max(0, layer_gpu_waittime),  # Ensure non-negative wait time
            "Layer Memcpy Bytes": layer_memcpy_bytes,
            "Kernel Inference Time": json.dumps(kernel_inference),
            "Kernel Memcpy Bytes": json.dumps(memcpy_bytes)
        })
    
    return mapping


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pPerf_nsys.py <filename_prefix>")
        sys.exit(1)
    
    prefix = sys.argv[1]
    json_file_path = f"{prefix}.json"
    output_csv_path = f"{prefix}_mapping.csv"
    
    nvtx_events, cuda_events, pid_map, trace_process_events = process_json(json_file_path)
    fill_pending_inputs(nvtx_events)

    for pid, models in pid_map.items():
        if len(models) > 1:
            print(f"Warning: GlobalTid {pid} is associated with multiple models: {models}")
    
    nvtx_df = pd.DataFrame(nvtx_events)
    cuda_df = pd.DataFrame(cuda_events)
    mapping = generate_mapping(nvtx_df, cuda_df, trace_process_events)
    
    pd.DataFrame(mapping).to_csv(output_csv_path, index=False)
    print(f"Mapping saved to {output_csv_path}")


