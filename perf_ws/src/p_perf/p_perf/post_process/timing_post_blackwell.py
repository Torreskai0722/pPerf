import json
import bisect
import pandas as pd
import os
from tqdm import tqdm
from p_perf.general_utils import get_closest_token_from_timestamp, build_channel_timestamp_token_map
from p_perf.config.constant import image_models, lidar_models
from collections import Counter, defaultdict


class timing_processor:
    COPYKIND_MAPPING = {1: "host2device", 8: "device2device", 2: "device2host"}

    def __init__(self, nusc, raw_json, output_dir, index, scene, mode=None, publish_mode="bag"):
        self.raw_json = raw_json
        self.output_dir = output_dir
        self.index = index
        self.mode = mode
        self.publish_mode = publish_mode
        self.scene_token = scene
        
        # Build reverse mapping from model names to their types
        self._build_model_type_mapping()

        self.nvtx_events = []
        self.cuda_events = []
        self.pid_map = {}
        self.kernel_name_list = []
        self.trace_process_events = {}
        self.nvtx_df = None
        self.cuda_df = None

        # For correlation ID uniqueness checking
        self.correlation_id_counter = Counter()
        self.trace_correlation_id_counter = Counter()
        # Track PID/TID and kernel details for each correlation ID
        self.correlation_id_details = {}  # {correlation_id: [(pid, tid, kernel_name, start, end), ...]}
        self.trace_correlation_id_details = {}  # {correlation_id: [(pid, tid, start), ...]}

        # Initialize token maps for bag mode
        if self.publish_mode == "bag":
            self.lidar_token_map = build_channel_timestamp_token_map(nusc, self.scene_token, "LIDAR_TOP")
            self.image_token_map = build_channel_timestamp_token_map(nusc, self.scene_token, "CAM_FRONT")
            self._lidar_sorted_ts = sorted(self.lidar_token_map.keys())
            self._image_sorted_ts = sorted(self.image_token_map.keys())
        else:
            self._lidar_sorted_ts = self._image_sorted_ts = []

        self.lidar_model_names = [model[0] for model in lidar_models]

        print("publish_mode: ", self.publish_mode)

    def _get_closest_token_fast(self, timestamp: float, channel: str) -> str:
        """O(log n) lookup using bisect; use in bag mode when maps are large."""
        if not self._lidar_sorted_ts and not self._image_sorted_ts:
            return get_closest_token_from_timestamp(timestamp, self.lidar_token_map if channel == "lidar" else self.image_token_map)
        ts_list = self._lidar_sorted_ts if channel == "lidar" else self._image_sorted_ts
        token_map = self.lidar_token_map if channel == "lidar" else self.image_token_map
        i = bisect.bisect_left(ts_list, timestamp)
        if i == 0:
            return token_map[ts_list[0]]
        if i >= len(ts_list):
            return token_map[ts_list[-1]]
        t_lo, t_hi = ts_list[i - 1], ts_list[i]
        closest_ts = t_hi if (t_hi - timestamp) < (timestamp - t_lo) else t_lo
        return token_map[closest_ts]

    def _build_model_type_mapping(self):
        """Build mapping from model names to their types (lidar vs non-lidar)."""
        self.model_type_mapping = {}
        
        # Add lidar models (these are tuples, extract the first element which is the model name)
        for model_tuple in lidar_models:
            model_name = model_tuple[0]  # Extract model name from (model_name, dataset, threshold)
            self.model_type_mapping[model_name] = 'lidar'

    @staticmethod
    def decode_globalid(global_id):
        PID = (global_id >> 24) & 0xFFFFFF
        TID = global_id & 0xFFFFFF
        return PID, TID

    def _process_data_preprocessor_layer(self, model_name, layer_parts):
        """
        Process individual data_preprocessor layers based on model type.
        
        Args:
            model_name: Name of the model
            layer_parts: List of layer parts from NVTX text
            
        Returns:
            Combined layer name and whether it should be processed
        """
        # Check if this is a data_preprocessor layer
        if len(layer_parts) >= 2 and layer_parts[0] == 'data_preprocessor':
            model_type = self.model_type_mapping.get(model_name, 'non-lidar')  # Default to non-lidar
            
            if model_type == 'lidar':
                # For lidar models, only include pipeline_{i} layers
                if len(layer_parts) >= 2 and 'pipeline_' in layer_parts[1]:
                    return 'data_preprocessing'
            else:
                # For non-lidar models, combine all data_preprocessor layers into one
                return 'data_preprocessing'
        
        # Not a data_preprocessor pipeline for lidar or not a data_preprocessor layer, keep original
        return '.'.join(layer_parts)

    def process_nvtx_event(self, entry):
        nvtx = entry["NvtxEvent"]
        if entry['Type'] == 59 and "Text" not in nvtx:
            return
        if "CCCL" in nvtx["Text"]:
            return
        try:
            parts = nvtx["Text"].split('.')
        except:
            print(f"Error splitting NVTX text: {nvtx['Text']}")
        if parts[-1] == 'e2e':
            if self.publish_mode == "bag" and len(parts) >= 3:
                model_name = parts[2]
                input_name = '.'.join(parts[:2])  # Join first two parts for input name (timestamp)
                try:
                    sec = float(input_name)
                    if model_name in self.lidar_model_names:
                        input_name = self._get_closest_token_fast(sec, "lidar")
                    elif model_name in image_models:
                        input_name = self._get_closest_token_fast(sec, "image")
                except (ValueError, TypeError):
                    raise ValueError(f"Error parsing input name for E2E: {input_name}")
                layer = parts[3:]
            elif self.publish_mode != "bag":
                input_name, model_name, *layer = parts
            else:
                model_name, *layer = parts
                input_name = "pending"
        elif parts[-1] in ("image_decode", "lidar_decode"):
            # Only our inferencer decode ranges; ignore e.g. encode_decode from other sources
            layer = parts[-1]
            input_name = '.'.join(parts[:2])  # Join first two parts for input name (timestamp)
            try:
                sec = float(input_name)
                model_name = "pending"
                if layer == "lidar_decode":
                    input_name = self._get_closest_token_fast(sec, "lidar")
                else:
                    input_name = self._get_closest_token_fast(sec, "image")
            except (ValueError, TypeError):
                print(f"DECODE Error parsing input name: {input_name}")
                raise ValueError(f"Error parsing input name for DECODE: {input_name}")
        else:
            model_name, *layer = parts
            input_name = "pending"

        # Process data_preprocessor layers
        if layer and isinstance(layer, list) and len(layer) > 0:
            combined_layer = self._process_data_preprocessor_layer(model_name, layer)
            layer = combined_layer
        elif isinstance(layer, str):
            # Handle case where layer is already a string
            layer_parts = layer.split('.')
            combined_layer = self._process_data_preprocessor_layer(model_name, layer_parts)
            layer = combined_layer

        pid, tid = self.decode_globalid(int(nvtx["GlobalTid"]))
        id_to_use = tid if self.mode == 'ms' else pid
        start = int(nvtx["Timestamp"])
        if "EndTimestamp" not in nvtx:
            print(f"NVTX event missing EndTimestamp (treating as instant): Text={nvtx.get('Text', '?')!r}, Timestamp={start}")
            return
        end = int(nvtx["EndTimestamp"])

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
        # Index by (pid, model) so we only scan relevant e2e events per pending event
        by_pid_model = defaultdict(list)
        for e in input_events:
            by_pid_model[(e["PID"], e["Model Name"])].append(e)
        for event in self.nvtx_events:
            if event["Input"] != "pending":
                continue
            pid, model, start, end = event["PID"], event["Model Name"], event["StartTimestamp"], event["EndTimestamp"]
            candidates = [
                e for e in by_pid_model[(pid, model)]
                if e["StartTimestamp"] <= start <= e["EndTimestamp"] and e["StartTimestamp"] <= end <= e["EndTimestamp"]
            ]
            if candidates:
                event["Input"] = min(candidates, key=lambda x: x["StartTimestamp"])["Input"]

    def process_cuda_event(self, entry):
        cuda = entry["CudaEvent"]
        eventClass = cuda.get("eventClass")
        pid, tid = self.decode_globalid(int(cuda["globalPid"]))
        id_to_use = tid if self.mode == 'ms' else pid
        memcpy_size = 0

        if eventClass == 3:  # Kernel
            kernel_index = int(cuda["kernel"].get("demangledName"))
            kernel_name = self.kernel_name_list[kernel_index]
        elif eventClass == 1:  # Memcpy
            mem_cpy = cuda.get("memcpy")
            kernel_name = self.COPYKIND_MAPPING.get(int(mem_cpy.get("copyKind")))
            memcpy_size = int(mem_cpy.get("sizebytes", 0))
        else:
            return

        # Count correlation ID for uniqueness check
        correlation_id = cuda.get("correlationId")
        if correlation_id is not None:
            self.correlation_id_counter[correlation_id] += 1
            # Track PID/TID and kernel details for this correlation ID
            kernel_start = int(cuda.get("startNs", 0))
            kernel_end = int(cuda.get("endNs", 0))
            if correlation_id not in self.correlation_id_details:
                self.correlation_id_details[correlation_id] = []
            self.correlation_id_details[correlation_id].append((pid, tid, kernel_name, kernel_start, kernel_end))

        self.cuda_events.append({
            "Kernel Name": kernel_name,
            "Kernel Start": int(cuda.get("startNs", 0)),
            "Kernel End": int(cuda.get("endNs", 0)),
            "Kernel Elapsed": (int(cuda.get("endNs", 0)) - int(cuda.get("startNs", 0))) * 1e-6,
            "Memcpy Size": memcpy_size,
            "CorrelationId": correlation_id,
            "PID": id_to_use,
            "OriginalPID": pid,
            "OriginalTID": tid
        })

    def process_trace_event(self, entry):
        trace = entry["TraceProcessEvent"]
        pid, tid = self.decode_globalid(int(trace["globalTid"]))
        id_to_use = tid if self.mode == 'ms' else pid
        correlation_id = trace["correlationId"]
        self.trace_process_events[correlation_id] = (int(trace["startNs"]), id_to_use)
        # Count correlation ID for uniqueness check
        if correlation_id is not None:
            self.trace_correlation_id_counter[correlation_id] += 1
            # Track PID/TID and start time for this correlation ID
            start_time = int(trace["startNs"])
            if correlation_id not in self.trace_correlation_id_details:
                self.trace_correlation_id_details[correlation_id] = []
            self.trace_correlation_id_details[correlation_id].append((pid, tid, start_time))

    def check_correlation_id_uniqueness(self):
        """
        Checks if correlation IDs are unique within each process for CUDA and TraceProcess events.
        Only checks PIDs that appear in NVTX events since those are the only ones relevant for correlation.
        """
        # Get PIDs that appear in NVTX events - these are the only ones we care about
        nvtx_pids = set(self.nvtx_df["PID"].unique()) if self.nvtx_df is not None and not self.nvtx_df.empty else set()
        
        # Check for duplicates within the same process (only for NVTX PIDs)
        cuda_process_duplicates = {}
        trace_process_duplicates = {}
        
        # Group CUDA correlation IDs by process (only for NVTX PIDs)
        cuda_by_process = {}
        for cid, details in self.correlation_id_details.items():
            for pid, tid, kernel_name, start, end in details:
                if pid in nvtx_pids:  # Only check PIDs that appear in NVTX events
                    if pid not in cuda_by_process:
                        cuda_by_process[pid] = Counter()
                    cuda_by_process[pid][cid] += 1
        
        # Find duplicates within each process
        for pid, counter in cuda_by_process.items():
            duplicates = {cid: count for cid, count in counter.items() if count > 1}
            if duplicates:
                cuda_process_duplicates[pid] = duplicates
        
        # Get all (correlation_id, pid) pairs that appear in CUDA kernel events
        cuda_correlation_pid_pairs = set()
        for cid, details in self.correlation_id_details.items():
            for pid, tid, kernel_name, start, end in details:
                cuda_correlation_pid_pairs.add((cid, pid))
        
        # Group TraceProcess correlation IDs by process (only for NVTX PIDs and only if they also appear in CUDA kernels from same PID)
        trace_by_process = {}
        for cid, details in self.trace_correlation_id_details.items():
            for pid, tid, start_time in details:
                if pid in nvtx_pids and (cid, pid) in cuda_correlation_pid_pairs:  # Only count if correlation ID and PID pair appears in CUDA kernels
                    if pid not in trace_by_process:
                        trace_by_process[pid] = Counter()
                    trace_by_process[pid][cid] += 1
        
        # Find duplicates within each process
        for pid, counter in trace_by_process.items():
            duplicates = {cid: count for cid, count in counter.items() if count > 1}
            if duplicates:
                trace_process_duplicates[pid] = duplicates
        
        # Report overall statistics
        total_cuda_duplicates = sum(len(dups) for dups in cuda_process_duplicates.values())
        total_trace_duplicates = sum(len(dups) for dups in trace_process_duplicates.values())
        
        print(f"Correlation ID uniqueness check (for NVTX PIDs: {sorted(nvtx_pids)}):")
        print(f"  CUDA: {total_cuda_duplicates} duplicate correlation IDs found across {len(cuda_process_duplicates)} processes")
        print(f"  TraceProcess: {total_trace_duplicates} duplicate correlation IDs found across {len(trace_process_duplicates)} processes")
        
                # Report detailed information for processes with duplicate correlation IDs
        if cuda_process_duplicates:
            print(f"\nWARNING: Found CUDA correlation ID duplicates within processes:")
            for pid, duplicates in cuda_process_duplicates.items():
                print(f"  Process {pid} has duplicate correlation IDs: {duplicates}")
                for cid in duplicates.keys():
                    details = [d for d in self.correlation_id_details[cid] if d[0] == pid]
                    print(f"    Correlation ID {cid}:")
                    for i, (_, tid, kernel_name, start, end) in enumerate(details):
                        print(f"      [{i+1}] TID: {tid}, Kernel: {kernel_name}, Start: {start}, End: {end}")
        else:
            print("All CUDA correlation IDs are unique within their respective processes.")
             
        if trace_process_duplicates:
            print(f"\nWARNING: Found TraceProcess correlation ID duplicates within processes:")
            for pid, duplicates in trace_process_duplicates.items():
                print(f"  Process {pid} has duplicate correlation IDs: {duplicates}")
                for cid in duplicates.keys():
                    details = [d for d in self.trace_correlation_id_details[cid] if d[0] == pid]
                    print(f"    Correlation ID {cid}:")
                    for i, (_, tid, start_time) in enumerate(details):
                        print(f"      [{i+1}] TID: {tid}, Start: {start_time}")
        else:
            print("All TraceProcess correlation IDs are unique within their respective processes.")

    def parse_json(self, logging=False):
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
        # Check correlation ID uniqueness after parsing
        if logging:
            self.check_correlation_id_uniqueness()

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

    def _combine_data_preprocessor_layers(self, layer_records):
        """
        Combine data_preprocessor layers that belong to the same e2e layer (same input/model).
        Replaces original layers with combined ones.
        
        Args:
            layer_records: List of layer records
            kernel_records: List of kernel records (not used in combining)
            
        Returns:
            Tuple of (combined layer records, original kernel records)
        """
        combined_layer_records = []
        
        # Group data_preprocessing events by input and model
        data_preprocessing_groups = {}
        other_events = []
        
        for record in layer_records:
            if record["Layer"] == "data_preprocessing":
                # Group by input and model combination
                key = (record["Input"], record["Model"])
                if key not in data_preprocessing_groups:
                    data_preprocessing_groups[key] = []
                data_preprocessing_groups[key].append(record)
            else:
                other_events.append(record)
        
        # Add all non-data_preprocessing events
        combined_layer_records.extend(other_events)
        
        # Add combined data_preprocessing records for each group
        for (input_name, model_name), events in data_preprocessing_groups.items():
            # Find the overall time range
            all_starts = [event["Start Timestamp"] for event in events]
            all_ends = [event["End Timestamp"] for event in events]
            
            combined_start = min(all_starts)
            combined_end = max(all_ends)
            
            # Sum up elapsed time only
            total_elapsed = sum(event["Elapsed Time"] for event in events)
            
            # Create combined record with only essential fields
            combined_layer_record = {
                "Input": input_name,
                "Model": model_name,
                "Layer": "data_preprocessing",
                "Start Timestamp": combined_start,
                "End Timestamp": combined_end,
                "Elapsed Time": total_elapsed,
                "GPU Turnaround Time": 0,
                "GPU Computation Time": 0,
                "GPU Wait Time": 0,
                "Internal Memcpy Size": 0,
                "External Memcpy Size": 0
            }
            
            combined_layer_records.append(combined_layer_record)
        
        # Return original kernel records unchanged (they won't match the combined layers anyway)
        return combined_layer_records

    def _build_trace_and_cuda_index(self):
        """Pre-index trace by pid (sorted by ts) and cuda by (pid, cid) for fast per-layer lookup."""
        trace_by_pid = defaultdict(list)
        for cid, (ts, p) in self.trace_process_events.items():
            trace_by_pid[p].append((ts, cid))
        for p in trace_by_pid:
            trace_by_pid[p].sort(key=lambda x: x[0])

        cuda_by_pid_cid = defaultdict(list)
        if self.cuda_df.empty:
            return trace_by_pid, cuda_by_pid_cid
        # itertuples uses underscored names (spaces -> _)
        for row in self.cuda_df.itertuples(index=False):
            cid = getattr(row, "CorrelationId", None)
            if cid is None:
                continue
            pid = getattr(row, "PID", None)
            cuda_by_pid_cid[(pid, cid)].append({
                "Kernel Start": getattr(row, "Kernel_Start", None),
                "Kernel End": getattr(row, "Kernel_End", None),
                "Kernel Name": getattr(row, "Kernel_Name", None),
                "Kernel Elapsed": getattr(row, "Kernel_Elapsed", None),
                "Memcpy Size": getattr(row, "Memcpy_Size", 0),
            })
        return trace_by_pid, cuda_by_pid_cid

    @staticmethod
    def _compute_gpu_time_from_list(candidate_list):
        """Same as compute_gpu_time but for list of kernel dicts with 'Kernel Start'/'Kernel End'."""
        if not candidate_list:
            return 0
        intervals = sorted((k["Kernel Start"], k["Kernel End"]) for k in candidate_list)
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

        trace_by_pid, cuda_by_pid_cid = self._build_trace_and_cuda_index()

        nvtx_cols = list(self.nvtx_df.columns)
        nvtx_tuples = list(self.nvtx_df.itertuples(index=False, name=None))

        for row in tqdm(nvtx_tuples, desc="Processing NVTX events"):
            nvtx = dict(zip(nvtx_cols, row))
            pid = nvtx["PID"]
            start = nvtx["StartTimestamp"]
            end = nvtx["EndTimestamp"]

            # TraceProcessEvents in [start, end] for this pid (bisect on sorted ts list)
            trace_list = trace_by_pid.get(pid, [])
            trace_cids = {}
            if trace_list:
                ts_only = [x[0] for x in trace_list]
                lo = bisect.bisect_left(ts_only, start)
                hi = bisect.bisect_right(ts_only, end)
                trace_cids = {cid: ts for ts, cid in trace_list[lo:hi]}

            candidate_list = []
            for cid in trace_cids:
                for k in cuda_by_pid_cid.get((pid, cid), []):
                    kstart, kend = k["Kernel Start"], k["Kernel End"]
                    if kstart >= start and kend <= end:
                        candidate_list.append(k)

            if candidate_list:
                k_starts = [k["Kernel Start"] for k in candidate_list]
                k_ends = [k["Kernel End"] for k in candidate_list]
                layer_gpu_turnaround = (max(k_ends) - min(k_starts)) * 1e-6
            else:
                layer_gpu_turnaround = 0
            gpu_active_time = self._compute_gpu_time_from_list(candidate_list)
            layer_gpu_waittime = layer_gpu_turnaround - gpu_active_time
            layer_cpu_time = (end - start) * 1e-6

            internal_memcpy = sum(k["Memcpy Size"] for k in candidate_list if k["Kernel Name"] == "device2device")
            external_memcpy = sum(k["Memcpy Size"] for k in candidate_list if k["Kernel Name"] in ("host2device", "device2host"))

            for k in candidate_list:
                kernel_records.append({
                    "Input": nvtx["Input"],
                    "Model": nvtx["Model Name"],
                    "Layer": nvtx["Layer"],
                    "Kernel Name": k["Kernel Name"],
                    "Start Timestamp": k["Kernel Start"],
                    "End Timestamp": k["Kernel End"],
                    "Elapsed Time": k["Kernel Elapsed"],
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

        # DEBUG: Check individual data_preprocessing records before combining
        debug_data_prep = [r for r in layer_records if r["Layer"] == "data_preprocessing"]
        if debug_data_prep:
            print("\n=== DEBUG: Individual data_preprocessing records (before combining) ===")
            for i, rec in enumerate(debug_data_prep[:5]):  # Show first 5
                elapsed_from_timestamps = (rec["End Timestamp"] - rec["Start Timestamp"]) * 1e-6
                print(f"Record {i}: Model={rec['Model']}, Input={rec['Input'][:30]}...")
                print(f"  Start: {rec['Start Timestamp']}, End: {rec['End Timestamp']}")
                print(f"  Elapsed (stored): {rec['Elapsed Time']:.6f} ms")
                print(f"  Elapsed (calc from timestamps): {elapsed_from_timestamps:.6f} ms")
                print(f"  MATCH: {abs(rec['Elapsed Time'] - elapsed_from_timestamps) < 0.001}")
        
        # Combine data_preprocessor layers
        layer_records = self._combine_data_preprocessor_layers(layer_records)

        if saving:
            self.save_results(layer_records, kernel_records, self.output_dir)
        return layer_records, kernel_records

    def save_results(self, layer_records, kernel_records, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(layer_records).to_csv(os.path.join(output_dir, f"layer_timings_{self.index}.csv"), index=False)
        pd.DataFrame(kernel_records).to_csv(os.path.join(output_dir, f"kernel_timings_{self.index}.csv"), index=False)

    def cleanup(self):
        """Release memory after processing is complete."""
        self.nvtx_events.clear()
        self.cuda_events.clear()
        self.kernel_name_list.clear()
        self.pid_map.clear()
        self.trace_process_events.clear()
        self.model_type_mapping.clear()
        self.correlation_id_counter.clear()
        self.trace_correlation_id_counter.clear()
        self.correlation_id_details.clear()
        self.trace_correlation_id_details.clear()
        
        # Clear DataFrames
        if hasattr(self, 'nvtx_df'):
            del self.nvtx_df
        if hasattr(self, 'cuda_df'):
            del self.cuda_df