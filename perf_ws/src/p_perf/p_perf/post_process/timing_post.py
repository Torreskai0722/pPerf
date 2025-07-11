import json
import pandas as pd
import os
from tqdm import tqdm
from p_perf.utils import get_closest_token_from_timestamp, build_channel_timestamp_token_map
from p_perf.nuscenes_instance import get_nuscenes_instance
from p_perf.config.constant import image_models, lidar_models


class timing_processor:
    COPYKIND_MAPPING = {1: "host2device", 8: "device2device", 2: "device2host"}

    def __init__(self, nusc, raw_json, output_dir, index, scene, mode=None, publish_mode="bag"):
        self.raw_json = raw_json
        self.output_dir = output_dir
        self.index = index
        self.mode = mode
        self.publish_mode = publish_mode
        self.scene_token = scene

        self.nvtx_events = []
        self.cuda_events = []
        self.pid_map = {}
        self.kernel_name_list = []
        self.trace_process_events = {}
        self.nvtx_df = None
        self.cuda_df = None

        # Initialize token maps for bag mode
        if self.publish_mode == "bag":
            self.lidar_token_map = build_channel_timestamp_token_map(nusc, self.scene_token, "LIDAR_TOP")
            self.image_token_map = build_channel_timestamp_token_map(nusc, self.scene_token, "CAM_FRONT")
        
        self.lidar_model_names = [model[0] for model in lidar_models]

    @staticmethod
    def decode_globalid(global_id):
        PID = (global_id >> 24) & 0xFFFFFF
        TID = global_id & 0xFFFFFF
        return PID, TID

    def process_nvtx_event(self, entry):
        nvtx = entry["NvtxEvent"]
        parts = nvtx["Text"].split('.')
        if parts[-1] == 'e2e':
            if self.publish_mode == "bag":
                model_name = parts[2]
                input_name = '.'.join(parts[:2])  # Join first two parts for input name
                sec = float(input_name)  # Convert input name to timestamp
                if model_name in self.lidar_model_names:
                    input_name = get_closest_token_from_timestamp(sec, self.lidar_token_map)
                elif model_name in image_models:
                    input_name = get_closest_token_from_timestamp(sec, self.image_token_map)
                layer = parts[3:]
            else:
                input_name, model_name, *layer = parts
        elif 'decode' in parts[-1]:
            layer = parts[-1]
            input_name = '.'.join(parts[:2])  # Join first two parts for input name
            sec = float(input_name)  # Convert input name to timestamp
            model_name = "pending"
            if "lidar" in layer:
                input_name = get_closest_token_from_timestamp(sec, self.lidar_token_map)
            elif "image" in layer:
                input_name = get_closest_token_from_timestamp(sec, self.image_token_map)
        else:
            model_name, *layer = parts
            input_name = "pending"

        pid, tid = self.decode_globalid(int(nvtx["GlobalTid"]))
        id_to_use = tid if self.mode == 'ms' else pid
        start, end = int(nvtx["Timestamp"]), int(nvtx["EndTimestamp"])
        
        self.nvtx_events.append({
            "Model Name": model_name,
            "Input": input_name,
            "Layer": layer if isinstance(layer, str) else '.'.join(layer),
            "StartTimestamp": start,
            "EndTimestamp": end,
            "Elapsed": (end - start) * 1e-6,
            "PID": id_to_use,
            "OriginalPID": pid,
            "OriginalTID": tid
        })
        self.pid_map.setdefault(id_to_use, set()).add(model_name)
        

    def fill_pending_inputs(self):
        input_events = [e for e in self.nvtx_events if e["Input"] != "pending"]
        for event in self.nvtx_events:
            if event["Input"] != "pending":
                continue
            pid, model, start, end = event["PID"], event["Model Name"], event["StartTimestamp"], event["EndTimestamp"]
            candidates = [
                e for e in input_events
                if e["PID"] == pid and e["Model Name"] == model and
                e["StartTimestamp"] <= start <= e["EndTimestamp"] and
                e["StartTimestamp"] <= end <= e["EndTimestamp"]
            ]
            if candidates:
                matched_input = sorted(candidates, key=lambda x: x["StartTimestamp"])
                event["Input"] = matched_input[0]["Input"]

    def process_cuda_event(self, entry):
        cuda = entry["CudaEvent"]
        eventClass = cuda.get("eventClass")
        pid, tid = self.decode_globalid(int(cuda["globalPid"]))
        id_to_use = tid if self.mode == 'ms' else pid
        memcpy_size = 0

        if eventClass == 3:  # Kernel
            kernel_index = int(cuda["kernel"].get("shortName"))
            kernel_name = self.kernel_name_list[kernel_index]
        elif eventClass == 1:  # Memcpy
            mem_cpy = cuda.get("memcpy")
            kernel_name = self.COPYKIND_MAPPING.get(int(mem_cpy.get("copyKind")))
            memcpy_size = int(mem_cpy.get("sizebytes", 0))
        else:
            return

        self.cuda_events.append({
            "Kernel Name": kernel_name,
            "Kernel Start": int(cuda.get("startNs", 0)),
            "Kernel End": int(cuda.get("endNs", 0)),
            "Kernel Elapsed": (int(cuda.get("endNs", 0)) - int(cuda.get("startNs", 0))) * 1e-6,
            "Memcpy Size": memcpy_size,
            "CorrelationId": cuda.get("correlationId"),
            "PID": id_to_use,
            "OriginalPID": pid,
            "OriginalTID": tid
        })

    def process_trace_event(self, entry):
        trace = entry["TraceProcessEvent"]
        pid, tid = self.decode_globalid(int(trace["globalTid"]))
        id_to_use = tid if self.mode == 'ms' else pid
        self.trace_process_events[trace["correlationId"]] = (int(trace["startNs"]), id_to_use)

    def parse_json(self):
        with open(self.raw_json, "r") as file:
            for i, line in enumerate(file):
                try:
                    entry = json.loads(line.strip())
                    if i == 0 and "data" in entry:
                        self.kernel_name_list = entry["data"]
                        continue
                    if "NvtxEvent" in entry:
                        self.process_nvtx_event(entry)
                    elif "CudaEvent" in entry:
                        self.process_cuda_event(entry)
                    elif "TraceProcessEvent" in entry:
                        self.process_trace_event(entry)
                except json.JSONDecodeError:
                    continue
        self.fill_pending_inputs()
        self.nvtx_df = pd.DataFrame(self.nvtx_events)
        self.cuda_df = pd.DataFrame(self.cuda_events)

    @staticmethod
    def compute_gpu_time(candidate_kernels):
        if candidate_kernels.empty:
            return 0
        intervals = sorted(zip(candidate_kernels["Kernel Start"], candidate_kernels["Kernel End"]))
        total_time = 0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start > current_end:
                total_time += (current_end - current_start)
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        total_time += (current_end - current_start)
        return total_time * 1e-6

    def generate_mapping(self, saving=True):
        layer_records = []
        kernel_records = []

        for _, nvtx in tqdm(self.nvtx_df.iterrows(), desc="Processing NVTX events"):
            pid = nvtx["PID"]
            start = nvtx["StartTimestamp"]
            end = nvtx["EndTimestamp"]
            
            # Try correlation through TraceProcessEvents
            # Find TraceProcessEvents that occur within the NVTX time range and belong to the same PID
            trace_candidates = {
                cid: ts for cid, (ts, p) in self.trace_process_events.items()
                if start <= ts <= end and p == pid
            }

            # Find CUDA kernels that have matching correlation IDs
            # Note: We don't require PID matching between CUDA and NVTX events
            # Correlation works through correlation IDs, not PID matching
            candidate_kernels = self.cuda_df[
                self.cuda_df["CorrelationId"].isin(trace_candidates.keys())
            ]
            
            layer_gpu_turnaround = (candidate_kernels["Kernel End"].max() - candidate_kernels["Kernel Start"].min()) * 1e-6 if not candidate_kernels.empty else 0
            gpu_active_time = self.compute_gpu_time(candidate_kernels)
            layer_gpu_waittime = layer_gpu_turnaround - gpu_active_time
            layer_cpu_time = (end - start) * 1e-6

            memcpy_kernels = candidate_kernels[
                candidate_kernels["Kernel Name"].isin(self.COPYKIND_MAPPING.values())
            ]
            internal_memcpy = memcpy_kernels[
                memcpy_kernels["Kernel Name"] == "device2device"
            ]["Memcpy Size"].sum()
            external_memcpy = memcpy_kernels[
                memcpy_kernels["Kernel Name"].isin(["host2device", "device2host"])
            ]["Memcpy Size"].sum()

            for _, krow in candidate_kernels.iterrows():
                kernel_records.append({
                    "Input": nvtx["Input"],
                    "Model": nvtx["Model Name"],
                    "Layer": nvtx["Layer"],
                    "Kernel Name": krow["Kernel Name"],
                    "Start Timestamp": krow["Kernel Start"],
                    "End Timestamp": krow["Kernel End"],
                    "Elapsed Time": krow["Kernel Elapsed"],
                })

            layer_records.append({
                "Input": nvtx["Input"],
                "Model": nvtx["Model Name"],
                "Layer": nvtx["Layer"],
                "Start Timestamp": start,
                "End Timestamp": end,
                "Elapsed Time": layer_cpu_time,
                "GPU Turnaround Time": layer_gpu_turnaround,
                "GPU Computation Time": gpu_active_time,
                "GPU Wait Time": max(0, layer_gpu_waittime),
                "Internal Memcpy Size": internal_memcpy,
                "External Memcpy Size": external_memcpy
            })

        if saving:
            self.save_results(layer_records, kernel_records, self.output_dir)
        return layer_records, kernel_records

    def save_results(self, layer_records, kernel_records, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(layer_records).to_csv(os.path.join(output_dir, f"layer_timings_{self.index}.csv"), index=False)
        pd.DataFrame(kernel_records).to_csv(os.path.join(output_dir, f"kernel_timings_{self.index}.csv"), index=False)
