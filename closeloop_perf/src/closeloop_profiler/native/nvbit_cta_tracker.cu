#include "nvbit_cta_abi.h"
#include "nvbit_cta_tracker_logic.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <vector>

#include "nvbit.h"
#include "nvbit_tool.h"

namespace {

struct DeviceCtaAccumulator {
  unsigned long long entry_ns;
  unsigned long long exit_ns;
  unsigned long long grid_id;
  unsigned int sm_id;
  unsigned int exit_sm_id;
  unsigned int exited_lanes;
  unsigned int nsmid;
};

DeviceCtaAccumulator* g_records{};
std::uint32_t* g_bases{};
std::uint32_t* g_threads{};
std::uint64_t* g_calibration_globaltimer{};

struct InstrumentedFunction {
  CUcontext context{};
  CUfunction function{};

  bool operator==(const InstrumentedFunction& other) const {
    return context == other.context && function == other.function;
  }
};

struct InstrumentedFunctionHash {
  std::size_t operator()(const InstrumentedFunction& value) const {
    const auto context = reinterpret_cast<std::uintptr_t>(value.context);
    const auto function = reinterpret_cast<std::uintptr_t>(value.function);
    return std::hash<std::uintptr_t>{}(context) ^
           (std::hash<std::uintptr_t>{}(function) << 1);
  }
};

struct PassiveLaunch {
  std::string target_label;
  std::string kernel_name;
  std::string driver_launch_api;
  std::uint32_t launch_slot{};
  std::uint32_t launch_sequence_index{};
  std::uint32_t occurrence_within_signature{};
  std::uint64_t context_handle{};
  std::uint64_t stream_handle{};
  std::uint64_t function_handle{};
  std::uint64_t host_launch_ready_ns{};
  std::uint32_t grid[3]{};
  std::uint32_t block[3]{};
  std::uint32_t dynamic_shared_memory{};
  int registers_per_thread{};
  int static_shared_memory{};
  std::uint64_t expected_count{};
  bool storage_reserved{};
};

std::mutex g_mutex;
std::unordered_set<InstrumentedFunction, InstrumentedFunctionHash>
    g_instrumented;
std::unordered_set<InstrumentedFunction, InstrumentedFunctionHash>
    g_enabled;
pperf_nvbit_cta_launch_spec g_launches[pperf::kNvbitCtaMaxLaunches];
std::size_t g_launch_count{};
std::size_t g_record_count{};
std::size_t g_record_capacity{pperf::kNvbitCtaMaxRecords};
std::uint64_t g_expected_record_count{};
std::uint64_t g_dropped_capacity_record_count{};
std::uint64_t g_dropped_launch_count{};
std::atomic<int> g_next_slot{-1};
std::atomic<int> g_error{};
std::atomic<bool> g_active{};

bool g_standalone{};
bool g_standalone_written{};
std::string g_standalone_model;
std::string g_standalone_output;
std::string g_standalone_name;
std::uint32_t g_standalone_grid[3]{};
std::uint32_t g_standalone_block[3]{};
std::vector<std::uint64_t> g_target_occurrences;
std::vector<std::string> g_target_labels;
std::vector<bool> g_target_observed;
std::vector<std::uint64_t> g_target_contexts;
std::vector<std::uint64_t> g_target_streams;
std::vector<std::uint64_t> g_target_functions;
std::vector<std::string> g_aligned_labels;
std::vector<std::string> g_barrier_participants;
std::vector<std::uint64_t> g_barrier_wait_ns;
std::vector<bool> g_barrier_complete;
std::uint64_t g_matching_occurrence{};
std::atomic<int> g_armed_slot{-1};
std::uint64_t g_within_input_occurrence{};
std::uint64_t g_calibration_host_before_ns{};
std::uint64_t g_calibration_host_after_ns{};
std::uint64_t g_calibration_timer_ns{};

bool g_passive{};
bool g_mixed{};
std::vector<std::uint32_t> g_passive_sequence_starts;
std::vector<std::uint32_t> g_passive_sequence_ends;
std::vector<std::uint32_t> g_passive_observed_sequences;
std::vector<bool> g_passive_disarmed;
std::vector<PassiveLaunch> g_passive_launches;
std::vector<std::unordered_set<InstrumentedFunction,
                               InstrumentedFunctionHash>>
    g_passive_sequence_functions;
std::unordered_set<InstrumentedFunction, InstrumentedFunctionHash>
    g_passive_predicted_functions;
std::unordered_map<InstrumentedFunction, std::uint32_t,
                   InstrumentedFunctionHash> g_passive_prearmed_slots;
std::vector<std::uint64_t> g_passive_predicted_function_counts;
std::vector<bool> g_passive_prediction_fallbacks;
std::unordered_map<std::string, std::uint32_t> g_passive_occurrences;
std::atomic<int> g_passive_target{-1};
std::atomic<bool> g_passive_learning{};
std::uint32_t g_passive_sequence{};
std::uint64_t g_passive_learned_inputs{};
std::uint64_t g_passive_enable_transitions{};
std::uint64_t g_passive_enable_refreshes{};
std::uint64_t g_passive_prepare_transitions{};
std::uint64_t g_passive_uninstrumentable_functions{};
std::uint64_t g_passive_disable_transitions{};

std::vector<std::string> split(const char* value) {
  std::vector<std::string> result;
  std::stringstream source(value == nullptr ? "" : value);
  std::string item;
  while (std::getline(source, item, ',')) result.push_back(item);
  return result;
}

bool parse_dimensions(const char* value, std::uint32_t result[3]) {
  const auto fields = split(value);
  if (fields.size() != 3) return false;
  for (std::size_t index = 0; index < 3; ++index) {
    char* end = nullptr;
    const auto parsed = std::strtoul(fields[index].c_str(), &end, 10);
    if (end == fields[index].c_str() || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::uint32_t>::max()) return false;
    result[index] = static_cast<std::uint32_t>(parsed);
  }
  return true;
}

bool parse_u32_list(const char* value, std::vector<std::uint32_t>& result) {
  result.clear();
  for (const auto& field : split(value)) {
    char* end = nullptr;
    const auto parsed = std::strtoull(field.c_str(), &end, 10);
    if (end == field.c_str() || *end != '\0' ||
        parsed > std::numeric_limits<std::uint32_t>::max()) return false;
    result.push_back(static_cast<std::uint32_t>(parsed));
  }
  return !result.empty();
}

std::string launch_signature_key(const std::string& name,
                                 const std::string& driver_launch_api,
                                 const std::uint32_t grid[3],
                                 const std::uint32_t block[3],
                                 std::uint32_t dynamic_shared_memory,
                                 int registers_per_thread,
                                 int static_shared_memory) {
  std::ostringstream value;
  value << name << '\n' << driver_launch_api << '\n'
        << grid[0] << ',' << grid[1] << ',' << grid[2]
        << '\n' << block[0] << ',' << block[1] << ',' << block[2]
        << '\n' << dynamic_shared_memory << '\n' << registers_per_thread
        << '\n' << static_shared_memory;
  return value.str();
}

std::uint64_t monotonic_ns() {
  timespec value{};
  clock_gettime(CLOCK_MONOTONIC, &value);
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

std::string json_string(const std::string& value) {
  std::string result = "\"";
  for (const char character : value) {
    if (character == '\\' || character == '"') result.push_back('\\');
    result.push_back(character);
  }
  return result + "\"";
}

bool contains(const std::vector<std::string>& values,
              const std::string& value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

void launch_barrier(std::size_t slot) {
  if (!contains(g_aligned_labels, g_target_labels[slot])) return;
  const auto separator = g_standalone_output.find_last_of('/');
  const auto root = g_standalone_output.substr(0, separator);
  const auto stem = root + "/cta_barrier_" + g_target_labels[slot] + "_";
  std::ofstream(stem + g_standalone_model + ".ready").put('\n');
  const auto start = monotonic_ns();
  const auto deadline = start + 5000000000ULL;
  bool complete = false;
  while (monotonic_ns() < deadline) {
    complete = std::all_of(
        g_barrier_participants.begin(), g_barrier_participants.end(),
        [&stem](const std::string& participant) {
          std::ifstream ready(stem + participant + ".ready");
          return ready.good();
        });
    if (complete) break;
    usleep(1000);
  }
  g_barrier_wait_ns[slot] = monotonic_ns() - start;
  g_barrier_complete[slot] = complete;
  if (!complete && !g_error) g_error = 32;
}

bool launch_event(nvbit_api_cuda_t cbid) {
  return cbid == API_CUDA_cuLaunchKernel ||
         cbid == API_CUDA_cuLaunchKernel_ptsz ||
         cbid == API_CUDA_cuLaunchKernelEx ||
         cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
         cbid == API_CUDA_cuLaunchCooperativeKernel ||
         cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz;
}

const char* driver_launch_api(nvbit_api_cuda_t cbid) {
  switch (cbid) {
    case API_CUDA_cuLaunchKernel: return "cuLaunchKernel";
    case API_CUDA_cuLaunchKernel_ptsz: return "cuLaunchKernel_ptsz";
    case API_CUDA_cuLaunchKernelEx: return "cuLaunchKernelEx";
    case API_CUDA_cuLaunchKernelEx_ptsz: return "cuLaunchKernelEx_ptsz";
    case API_CUDA_cuLaunchCooperativeKernel:
      return "cuLaunchCooperativeKernel";
    case API_CUDA_cuLaunchCooperativeKernel_ptsz:
      return "cuLaunchCooperativeKernel_ptsz";
    default: return "unknown";
  }
}

struct LaunchView {
  CUfunction function{};
  CUstream stream{};
  std::uint32_t grid[3]{};
  std::uint32_t block[3]{};
  std::uint32_t dynamic_shared_memory{};
};

LaunchView launch_view(nvbit_api_cuda_t cbid, void* params) {
  LaunchView result;
  if (cbid == API_CUDA_cuLaunchKernelEx ||
      cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
    const auto* value = static_cast<cuLaunchKernelEx_params*>(params);
    result.function = value->f;
    result.stream = value->config->hStream;
    result.grid[0] = value->config->gridDimX;
    result.grid[1] = value->config->gridDimY;
    result.grid[2] = value->config->gridDimZ;
    result.block[0] = value->config->blockDimX;
    result.block[1] = value->config->blockDimY;
    result.block[2] = value->config->blockDimZ;
    result.dynamic_shared_memory = value->config->sharedMemBytes;
  } else {
    const auto* value = static_cast<cuLaunchKernel_params*>(params);
    result.function = value->f;
    result.stream = value->hStream;
    result.grid[0] = value->gridDimX;
    result.grid[1] = value->gridDimY;
    result.grid[2] = value->gridDimZ;
    result.block[0] = value->blockDimX;
    result.block[1] = value->blockDimY;
    result.block[2] = value->blockDimZ;
    result.dynamic_shared_memory = value->sharedMemBytes;
  }
  return result;
}

bool matches(const LaunchView& observed,
             const pperf_nvbit_cta_launch_spec& expected) {
  return reinterpret_cast<std::uint64_t>(observed.function) ==
             expected.function_handle &&
         observed.grid[0] == expected.grid_x &&
         observed.grid[1] == expected.grid_y &&
         observed.grid[2] == expected.grid_z &&
         observed.block[0] == expected.block_x &&
         observed.block[1] == expected.block_y &&
         observed.block[2] == expected.block_z;
}

bool instrument(CUcontext context, CUfunction function) {
  const InstrumentedFunction key{context, function};
  if (!g_instrumented.insert(key).second) return true;
  const auto& instructions = nvbit_get_instrs(context, function);
  if (instructions.empty()) {
    g_instrumented.erase(key);
    return false;
  }
  auto* entry = instructions.front();
  nvbit_insert_call(entry, "pperf_cta_entry", IPOINT_BEFORE);
  nvbit_add_call_arg_launch_val64(entry, 0);
  nvbit_add_call_arg_const_val64(
      entry, reinterpret_cast<std::uint64_t>(g_records));
  nvbit_add_call_arg_const_val64(
      entry, reinterpret_cast<std::uint64_t>(g_bases));
  for (auto* instruction : instructions) {
    if (std::strcmp(instruction->getOpcodeShort(), "EXIT") != 0) continue;
    nvbit_insert_call(instruction, "pperf_cta_exit", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instruction);
    nvbit_add_call_arg_launch_val64(instruction, 0);
    nvbit_add_call_arg_const_val64(
        instruction, reinterpret_cast<std::uint64_t>(g_records));
    nvbit_add_call_arg_const_val64(
        instruction, reinterpret_cast<std::uint64_t>(g_bases));
    nvbit_add_call_arg_const_val64(
        instruction, reinterpret_cast<std::uint64_t>(g_threads));
  }
  return true;
}

void prepare_passive_function(CUcontext context, CUfunction function) {
  const InstrumentedFunction key{context, function};
  if (g_instrumented.count(key)) return;
  if (!instrument(context, function)) {
    ++g_passive_uninstrumentable_functions;
    return;
  }
  nvbit_enable_instrumented(context, function, false, false);
  ++g_passive_prepare_transitions;
}

bool refresh_passive_launch(CUcontext context, CUfunction function,
                            std::uint64_t launch_slot) {
  const InstrumentedFunction key{context, function};
  if (!g_enabled.count(key)) return false;
  g_passive_prearmed_slots.erase(key);
  nvbit_set_at_launch(context, function, launch_slot);
  nvbit_enable_instrumented(context, function, true, false);
  ++g_passive_enable_refreshes;
  return true;
}

bool select_passive_launch(CUcontext context, CUfunction function,
                           std::uint64_t launch_slot) {
  const InstrumentedFunction key{context, function};
  const int target = g_passive_target.load();
  const bool fallback = target >= 0 &&
      static_cast<std::size_t>(target) < g_passive_prediction_fallbacks.size()
      && g_passive_prediction_fallbacks[target];
  if (!g_passive_predicted_functions.count(key) && !fallback) return false;
  const auto prearmed = g_passive_prearmed_slots.find(key);
  if (prearmed != g_passive_prearmed_slots.end()
      && prearmed->second == launch_slot && g_enabled.count(key)) {
    return true;
  }
  if (!g_instrumented.count(key) && !instrument(context, function)) {
    return false;
  }
  nvbit_set_at_launch(context, function, launch_slot);
  if (g_enabled.insert(key).second) {
    ++g_passive_enable_transitions;
  } else {
    ++g_passive_enable_refreshes;
  }
  nvbit_enable_instrumented(context, function, true, false);
  return true;
}

__global__ void calibrate_globaltimer(std::uint64_t* result) {
  std::uint64_t now = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  *result = now;
}

bool standalone_match(CUcontext context, const LaunchView& launch) {
  if (launch.grid[0] != g_standalone_grid[0] ||
      launch.grid[1] != g_standalone_grid[1] ||
      launch.grid[2] != g_standalone_grid[2] ||
      launch.block[0] != g_standalone_block[0] ||
      launch.block[1] != g_standalone_block[1] ||
      launch.block[2] != g_standalone_block[2]) return false;
  const char* observed = nvbit_get_func_name(context, launch.function);
  return observed != nullptr &&
         std::strstr(observed, g_standalone_name.c_str()) != nullptr;
}

void write_standalone();

}  // namespace

bool initialize_storage();

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 0);
  const char* output = std::getenv("PPERF_NVBIT_CTA_OUTPUT_PREFIX");
  const char* name = std::getenv("PPERF_NVBIT_CTA_TARGET_NAME");
  const char* mode = std::getenv("PPERF_NVBIT_CTA_MODE");
  g_mixed = mode != nullptr && std::strcmp(mode, "mixed") == 0;
  g_passive = g_mixed ||
      (mode != nullptr && std::strcmp(mode, "passive") == 0);
  const char* capacity = std::getenv("PPERF_NVBIT_CTA_RECORD_CAPACITY");
  if (capacity != nullptr) {
    char* end = nullptr;
    const auto parsed = std::strtoull(capacity, &end, 10);
    if (end == capacity || *end != '\0' || parsed == 0 ||
        parsed > pperf::kNvbitCtaMaxRecords) {
      g_error = 26;
      return;
    }
    g_record_capacity = static_cast<std::size_t>(parsed);
  }
  if (output == nullptr || (!g_passive && name == nullptr)) return;
  g_standalone = true;
  g_standalone_output = output;
  g_standalone_name = name == nullptr ? "*" : name;
  const char* model = std::getenv("PPERF_NVBIT_CTA_MODEL_ID");
  g_standalone_model = model == nullptr ? "unknown" : model;
  g_target_labels = split(std::getenv("PPERF_NVBIT_CTA_TARGET_LABELS"));
  if (g_passive) {
    if (!parse_u32_list(
            std::getenv("PPERF_NVBIT_CTA_SEQUENCE_STARTS"),
            g_passive_sequence_starts) ||
        !parse_u32_list(
            std::getenv("PPERF_NVBIT_CTA_SEQUENCE_ENDS"),
            g_passive_sequence_ends) ||
        g_target_labels.empty() ||
        g_target_labels.size() != g_passive_sequence_starts.size() ||
        g_target_labels.size() != g_passive_sequence_ends.size() ||
        std::any_of(
            g_passive_sequence_starts.begin(),
            g_passive_sequence_starts.end(),
            [](std::uint32_t value) { return value == 0; })) {
      g_error = 27;
      return;
    }
    for (std::size_t index = 0; index < g_target_labels.size(); ++index) {
      if (g_passive_sequence_starts[index] >
          g_passive_sequence_ends[index]) {
        g_error = 27;
        return;
      }
    }
    g_passive_observed_sequences.resize(g_target_labels.size());
    g_passive_disarmed.resize(g_target_labels.size());
    g_passive_predicted_function_counts.resize(g_target_labels.size());
    g_passive_prediction_fallbacks.resize(g_target_labels.size());
  }
  if (!g_passive || g_mixed) {
    if (!parse_dimensions(std::getenv("PPERF_NVBIT_CTA_TARGET_GRID"),
                          g_standalone_grid) ||
        !parse_dimensions(std::getenv("PPERF_NVBIT_CTA_TARGET_BLOCK"),
                          g_standalone_block)) {
      g_error = 20;
      return;
    }
    for (const auto& field : split(
             std::getenv("PPERF_NVBIT_CTA_TARGET_OCCURRENCES"))) {
      char* end = nullptr;
      const auto value = std::strtoull(field.c_str(), &end, 10);
      if (end == field.c_str() || *end != '\0') {
        g_error = 21;
        return;
      }
      g_target_occurrences.push_back(value);
    }
  } else {
    g_target_occurrences.resize(g_target_labels.size());
  }
  if (g_target_occurrences.empty() ||
      g_target_occurrences.size() != g_target_labels.size() ||
      g_target_occurrences.size() > pperf::kNvbitCtaMaxLaunches) {
    g_error = 22;
    return;
  }
  g_target_observed.resize(g_target_occurrences.size(), false);
  g_target_contexts.resize(g_target_occurrences.size());
  g_target_streams.resize(g_target_occurrences.size());
  g_target_functions.resize(g_target_occurrences.size());
  g_barrier_wait_ns.resize(g_target_occurrences.size());
  g_barrier_complete.resize(g_target_occurrences.size());
  g_aligned_labels = split(std::getenv("PPERF_NVBIT_CTA_ALIGN_LABELS"));
  g_barrier_participants = split(
      std::getenv("PPERF_NVBIT_CTA_BARRIER_PARTICIPANTS"));
}

void nvbit_at_term() { write_standalone(); }
void nvbit_at_ctx_init(CUcontext) {}
void nvbit_at_ctx_term(CUcontext) { write_standalone(); }
void nvbit_tool_init(CUcontext) {
  if (!g_standalone || g_error || !initialize_storage()) return;
  if (g_passive) {
    for (std::size_t index = 0; index < g_record_capacity; ++index) {
      g_records[index] = {
          std::numeric_limits<unsigned long long>::max(), 0,
          std::numeric_limits<unsigned long long>::max(),
          std::numeric_limits<unsigned int>::max(),
          std::numeric_limits<unsigned int>::max(), 0,
          std::numeric_limits<unsigned int>::max()};
    }
    return;
  }
  const std::uint64_t count =
      static_cast<std::uint64_t>(g_standalone_grid[0]) *
      g_standalone_grid[1] * g_standalone_grid[2];
  g_launch_count = g_target_occurrences.size();
  g_record_count = static_cast<std::size_t>(count * g_launch_count);
  g_expected_record_count = g_record_count;
  if (g_record_count > g_record_capacity) {
    g_error = 23;
    return;
  }
  for (std::size_t slot = 0; slot < g_launch_count; ++slot) {
    g_launches[slot] = {
        0, static_cast<std::uint32_t>(slot),
        g_standalone_grid[0], g_standalone_grid[1], g_standalone_grid[2],
        g_standalone_block[0], g_standalone_block[1], g_standalone_block[2]};
    g_bases[slot] = static_cast<std::uint32_t>(slot * count);
    g_threads[slot] = g_standalone_block[0] * g_standalone_block[1] *
                      g_standalone_block[2];
  }
  for (std::size_t index = 0; index < g_record_count; ++index) {
    g_records[index] = {
        std::numeric_limits<unsigned long long>::max(), 0,
        std::numeric_limits<unsigned long long>::max(),
        std::numeric_limits<unsigned int>::max(),
        std::numeric_limits<unsigned int>::max(), 0,
        std::numeric_limits<unsigned int>::max()};
  }
}
void nvbit_at_graph_node_launch(CUcontext, CUfunction, CUstream,
                                std::uint64_t) {}

bool initialize_storage() {
  if (g_records != nullptr) return true;
  if (cudaMallocManaged(
          reinterpret_cast<void**>(&g_records),
          sizeof(*g_records) * pperf::kNvbitCtaMaxRecords) != cudaSuccess ||
      cudaMallocManaged(
          reinterpret_cast<void**>(&g_bases),
          sizeof(*g_bases) * pperf::kNvbitCtaMaxLaunches) != cudaSuccess ||
      cudaMallocManaged(
          reinterpret_cast<void**>(&g_threads),
          sizeof(*g_threads) * pperf::kNvbitCtaMaxLaunches) != cudaSuccess ||
      cudaMallocManaged(
          reinterpret_cast<void**>(&g_calibration_globaltimer),
          sizeof(*g_calibration_globaltimer)) != cudaSuccess) {
    g_error = 12;
    return false;
  }
  return true;
}

namespace {

std::string csv_string(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
  std::string result = "\"";
  for (const char character : value) {
    if (character == '\"') result.push_back('\"');
    result.push_back(character);
  }
  return result + "\"";
}

void write_passive_files(std::int64_t offset, std::uint64_t clock_error) {
  const std::string csv_path = g_standalone_output + "_cta_raw.csv";
  const std::string csv_temporary = csv_path + ".tmp";
  std::ofstream csv(csv_temporary);
  csv << "model_id,pid,target_label,launch_slot,launch_sequence_index,"
         "occurrence_within_signature,kernel_name,driver_launch_api,"
         "launch_type,context_handle,"
         "stream_handle,function_handle,grid_x,grid_y,grid_z,block_x,"
         "block_y,block_z,dynamic_shared_memory,registers_per_thread,"
         "static_shared_memory,cta_id,cta_x,cta_y,cta_z,sm_id,exit_sm_id,"
         "sm_migrated,nsmid,grid_id,"
         "entry_globaltimer_ns,exit_globaltimer_ns,entry_ns,exit_ns,"
         "duration_ns,clock_error_ns,entry_observed,exit_observed,"
         "observed_exit_lane_count,observation_status\n";
  std::vector<std::uint64_t> entered(g_passive_launches.size());
  std::vector<std::uint64_t> exited(g_passive_launches.size());
  std::vector<std::uint64_t> complete(g_passive_launches.size());
  std::uint64_t incomplete_entries = 0;
  std::uint64_t incomplete_exits = 0;
  for (std::size_t index = 0; index < g_passive_launches.size(); ++index) {
    const auto& launch = g_passive_launches[index];
    if (!launch.storage_reserved) continue;
    for (std::uint64_t cta = 0; cta < launch.expected_count; ++cta) {
      const auto& record = g_records[g_bases[launch.launch_slot] + cta];
      const bool entry = record.entry_ns !=
          std::numeric_limits<unsigned long long>::max();
      const bool exit = record.exit_ns != 0;
      entered[index] += entry;
      exited[index] += exit;
      complete[index] += entry && exit;
      incomplete_entries += !entry;
      incomplete_exits += !exit;
      std::uint64_t entry_ns = 0;
      std::uint64_t exit_ns = 0;
      if (entry) pperf::calibrated_timestamp(record.entry_ns, offset,
                                             entry_ns);
      if (exit) pperf::calibrated_timestamp(record.exit_ns, offset, exit_ns);
      const auto cta_x = cta % launch.grid[0];
      const auto cta_y = (cta / launch.grid[0]) % launch.grid[1];
      const auto cta_z = cta /
          (static_cast<std::uint64_t>(launch.grid[0]) * launch.grid[1]);
      csv << csv_string(g_standalone_model) << ',' << getpid() << ','
          << csv_string(launch.target_label) << ',' << launch.launch_slot
          << ',' << launch.launch_sequence_index << ','
          << launch.occurrence_within_signature << ','
          << csv_string(launch.kernel_name) << ','
          << launch.driver_launch_api << ",kernel," << launch.context_handle
          << ',' << launch.stream_handle << ',' << launch.function_handle
          << ',' << launch.grid[0] << ',' << launch.grid[1] << ','
          << launch.grid[2] << ',' << launch.block[0] << ','
          << launch.block[1] << ',' << launch.block[2] << ','
          << launch.dynamic_shared_memory << ','
          << launch.registers_per_thread << ','
          << launch.static_shared_memory << ',' << cta << ',' << cta_x
          << ',' << cta_y << ',' << cta_z << ','
          << (entry ? record.sm_id : 0) << ','
          << (exit ? record.exit_sm_id : 0) << ','
          << (entry && exit && record.sm_id != record.exit_sm_id) << ','
          << (entry ? record.nsmid : 0) << ','
          << (entry ? record.grid_id : 0) << ','
          << (entry ? record.entry_ns : 0) << ','
          << (exit ? record.exit_ns : 0) << ',' << entry_ns << ','
          << exit_ns << ','
          << (entry && exit ? record.exit_ns - record.entry_ns : 0) << ','
          << clock_error << ',' << entry << ',' << exit << ','
          << record.exited_lanes << ','
          << (entry && exit ? "complete" : entry ? "missing_exit" :
              exit ? "missing_entry" : "missing") << '\n';
    }
  }
  csv.close();
  std::rename(csv_temporary.c_str(), csv_path.c_str());

  cudaDeviceProp device{};
  int device_index = 0;
  const bool device_available =
      cudaGetDevice(&device_index) == cudaSuccess &&
      cudaGetDeviceProperties(&device, device_index) == cudaSuccess;
  const std::string status_path = g_standalone_output + "_cta_status.json";
  const std::string status_temporary = status_path + ".tmp";
  std::ofstream status(status_temporary);
  status << "{\n  \"schema\": \"pperf_nvbit_cta_standalone_v2\",\n"
         << "  \"mode\": " << json_string(g_mixed ? "mixed" : "passive")
         << ",\n"
         << "  \"model_id\": " << json_string(g_standalone_model)
         << ",\n  \"pid\": " << getpid()
         << ",\n  \"tracker_version\": "
         << json_string(PPERF_NVBIT_CTA_VERSION)
         << ",\n  \"tracker_error\": " << g_error.load()
         << ",\n  \"launch_alignment_requested\": "
         << (g_aligned_labels.empty() ? "false" : "true") << ",\n"
         << "  \"launch_manipulation\": "
         << (g_aligned_labels.empty() ? "false" : "true") << ",\n"
         << "  \"instrumentation\": {\"activation\": "
            "\"learned_sequence_hybrid_pre_enable\", "
            "\"launch_value_gate\": true, "
            "\"learned_input_count\": "
         << g_passive_learned_inputs
         << ", \"learned_sequence_count\": "
         << (g_passive_sequence_functions.empty()
                 ? 0 : g_passive_sequence_functions.size() - 1)
         << ", "
            "\"enable_transitions\": "
         << g_passive_enable_transitions
         << ", \"enable_refreshes\": " << g_passive_enable_refreshes
         << ", \"prepare_transitions\": "
         << g_passive_prepare_transitions
         << ", \"uninstrumentable_functions\": "
         << g_passive_uninstrumentable_functions
         << ", \"disable_transitions\": "
         << g_passive_disable_transitions << "},\n"
         << "  \"collection\": {\"record_capacity\": "
         << g_record_capacity << ", \"record_high_water_mark\": "
         << g_record_count << ", \"expected_record_count\": "
         << g_expected_record_count << ", \"reserved_record_count\": "
         << g_record_count << ", \"dropped_capacity_record_count\": "
         << g_dropped_capacity_record_count
         << ", \"dropped_launch_count\": " << g_dropped_launch_count
         << ", \"incomplete_entry_count\": " << incomplete_entries
         << ", \"incomplete_exit_count\": " << incomplete_exits
         << "},\n  \"clock_calibration\": {\n"
         << "    \"host_before_ns\": " << g_calibration_host_before_ns
         << ",\n    \"host_after_ns\": " << g_calibration_host_after_ns
         << ",\n    \"globaltimer_ns\": " << g_calibration_timer_ns
         << ",\n    \"offset_ns\": " << offset
         << ",\n    \"error_ns\": " << clock_error
         << ",\n    \"cross_client_clock\": \"raw_gpu_globaltimer\""
         << "\n  },\n  \"device_limits\": ";
  if (!device_available) {
    status << "null";
  } else {
    status << "{\"compute_capability\": \"" << device.major << '.'
           << device.minor << "\", \"sm_count\": "
           << device.multiProcessorCount
           << ", \"max_threads_per_sm\": "
           << device.maxThreadsPerMultiProcessor
           << ", \"max_threads_per_block\": "
           << device.maxThreadsPerBlock << ", \"registers_per_sm\": "
           << device.regsPerMultiprocessor
           << ", \"shared_memory_per_sm\": "
           << device.sharedMemPerMultiprocessor
           << ", \"shared_memory_per_block_optin\": "
           << device.sharedMemPerBlockOptin
           << ", \"max_blocks_per_sm\": "
           << device.maxBlocksPerMultiProcessor
           << ", \"warp_size\": " << device.warpSize << '}';
  }
  status << ",\n  \"targets\": [\n";
  for (std::size_t target = 0; target < g_target_labels.size(); ++target) {
    if (target) status << ",\n";
    std::uint64_t expected = 0;
    std::uint64_t entered_count = 0;
    std::uint64_t exited_count = 0;
    std::uint64_t complete_count = 0;
    std::uint64_t captured_launches = 0;
    for (std::size_t index = 0; index < g_passive_launches.size(); ++index) {
      if (g_passive_launches[index].target_label != g_target_labels[target]) {
        continue;
      }
      ++captured_launches;
      expected += g_passive_launches[index].expected_count;
      entered_count += entered[index];
      exited_count += exited[index];
      complete_count += complete[index];
    }
    const auto expected_launches =
        g_passive_sequence_ends[target] -
        g_passive_sequence_starts[target] + 1;
    status << "    {\"label\": " << json_string(g_target_labels[target])
           << ", \"sequence_start\": "
           << g_passive_sequence_starts[target]
           << ", \"sequence_end\": " << g_passive_sequence_ends[target]
           << ", \"predicted_function_count\": "
           << g_passive_predicted_function_counts[target]
           << ", \"prediction_fallback\": "
           << (g_passive_prediction_fallbacks[target] ? "true" : "false")
           << ", \"observed_input_launch_count\": "
           << g_passive_observed_sequences[target]
           << ", \"disarmed\": "
           << (g_passive_disarmed[target] ? "true" : "false")
           << ", \"launch_alignment_requested\": "
           << (contains(g_aligned_labels, g_target_labels[target])
                   ? "true" : "false")
           << ", \"launch_alignment_complete\": "
           << (g_barrier_complete[target] ? "true" : "false")
           << ", \"launch_alignment_wait_ns\": "
           << g_barrier_wait_ns[target]
           << ", \"expected_launch_count\": " << expected_launches
           << ", \"captured_launch_count\": " << captured_launches
           << ", \"expected_count\": " << expected
           << ", \"entered_count\": " << entered_count
           << ", \"exited_count\": " << exited_count
           << ", \"complete_count\": " << complete_count
           << ", \"missing_count\": " << expected - complete_count
           << ", \"coverage_fraction\": "
           << (expected == 0 ? 0.0 : static_cast<double>(complete_count) /
                                           static_cast<double>(expected))
           << '}';
  }
  status << "\n  ],\n  \"launches\": [\n";
  for (std::size_t index = 0; index < g_passive_launches.size(); ++index) {
    const auto& launch = g_passive_launches[index];
    if (index) status << ",\n";
    status << "    {\"target_label\": "
           << json_string(launch.target_label)
           << ", \"launch_slot\": " << launch.launch_slot
           << ", \"launch_sequence_index\": "
           << launch.launch_sequence_index
           << ", \"occurrence_within_signature\": "
           << launch.occurrence_within_signature
           << ", \"kernel_name\": " << json_string(launch.kernel_name)
           << ", \"driver_launch_api\": "
           << json_string(launch.driver_launch_api)
           << ", \"launch_type\": \"kernel\""
           << ", \"context_handle\": " << launch.context_handle
           << ", \"stream_handle\": " << launch.stream_handle
           << ", \"function_handle\": " << launch.function_handle
           << ", \"host_launch_ready_ns\": "
           << launch.host_launch_ready_ns
           << ", \"launch_ready_globaltimer_ns\": "
           << static_cast<std::uint64_t>(
                  static_cast<std::int64_t>(launch.host_launch_ready_ns) -
                  offset)
           << ", \"launch_ready_error_ns\": " << clock_error
           << ", \"grid\": [" << launch.grid[0] << ',' << launch.grid[1]
           << ',' << launch.grid[2] << "], \"block\": ["
           << launch.block[0] << ',' << launch.block[1] << ','
           << launch.block[2] << "], \"dynamic_shared_memory\": "
           << launch.dynamic_shared_memory
           << ", \"registers_per_thread\": "
           << launch.registers_per_thread
           << ", \"static_shared_memory\": "
           << launch.static_shared_memory
           << ", \"expected_count\": " << launch.expected_count
           << ", \"entered_count\": " << entered[index]
           << ", \"exited_count\": " << exited[index]
           << ", \"complete_count\": " << complete[index]
           << ", \"dropped_capacity_count\": "
           << (launch.storage_reserved ? 0 : launch.expected_count)
           << ", \"storage_reserved\": "
           << (launch.storage_reserved ? "true" : "false") << '}';
  }
  status << "\n  ]\n}\n";
  status.close();
  std::rename(status_temporary.c_str(), status_path.c_str());
}

void write_standalone() {
  if (!g_standalone || g_standalone_written) return;
  g_standalone_written = true;
  if (g_records != nullptr && cudaDeviceSynchronize() != cudaSuccess &&
      !g_error) g_error = 24;
  if (g_records != nullptr) {
    g_calibration_host_before_ns = monotonic_ns();
    calibrate_globaltimer<<<1, 1>>>(g_calibration_globaltimer);
    if (cudaGetLastError() == cudaSuccess &&
        cudaDeviceSynchronize() == cudaSuccess) {
      g_calibration_host_after_ns = monotonic_ns();
      g_calibration_timer_ns = *g_calibration_globaltimer;
    } else if (!g_error) {
      g_error = 25;
    }
  }
  const auto midpoint = g_calibration_host_before_ns +
      (g_calibration_host_after_ns - g_calibration_host_before_ns) / 2;
  const auto offset = static_cast<std::int64_t>(midpoint) -
                      static_cast<std::int64_t>(g_calibration_timer_ns);
  const auto clock_error =
      (g_calibration_host_after_ns - g_calibration_host_before_ns) / 2;

  if (g_passive) {
    write_passive_files(offset, clock_error);
    return;
  }

  const std::string csv_path = g_standalone_output + "_cta_raw.csv";
  const std::string csv_temporary = csv_path + ".tmp";
  std::ofstream csv(csv_temporary);
  csv << "model_id,pid,target_label,launch_slot,target_occurrence,kernel_name,"
         "context_handle,stream_handle,function_handle,"
         "grid_x,grid_y,grid_z,block_x,block_y,block_z,cta_id,cta_x,cta_y,"
         "cta_z,sm_id,exit_sm_id,sm_migrated,nsmid,grid_id,"
         "entry_globaltimer_ns,"
         "exit_globaltimer_ns,entry_ns,"
         "exit_ns,duration_ns,clock_error_ns,entry_observed,exit_observed,"
         "observation_status\n";
  std::vector<std::uint64_t> entered(g_launch_count);
  std::vector<std::uint64_t> exited(g_launch_count);
  std::vector<std::uint64_t> complete(g_launch_count);
  const auto expected = static_cast<std::uint64_t>(g_standalone_grid[0]) *
                        g_standalone_grid[1] * g_standalone_grid[2];
  for (std::size_t slot = 0; slot < g_launch_count; ++slot) {
    for (std::uint64_t cta = 0; cta < expected; ++cta) {
      const auto& record = g_records[g_bases[slot] + cta];
      const bool entry = record.entry_ns !=
          std::numeric_limits<unsigned long long>::max();
      const bool exit = record.exit_ns != 0;
      entered[slot] += entry;
      exited[slot] += exit;
      complete[slot] += entry && exit;
      std::uint64_t entry_ns = 0;
      std::uint64_t exit_ns = 0;
      if (entry) pperf::calibrated_timestamp(
          record.entry_ns, offset, entry_ns);
      if (exit) pperf::calibrated_timestamp(record.exit_ns, offset, exit_ns);
      const auto cta_x = cta % g_standalone_grid[0];
      const auto cta_y =
          (cta / g_standalone_grid[0]) % g_standalone_grid[1];
      const auto cta_z = cta /
          (static_cast<std::uint64_t>(g_standalone_grid[0]) *
           g_standalone_grid[1]);
      csv << g_standalone_model << ',' << getpid() << ','
          << g_target_labels[slot] << ','
          << slot << ',' << g_target_occurrences[slot] << ','
          << g_standalone_name << ',' << g_target_contexts[slot] << ','
          << g_target_streams[slot] << ',' << g_target_functions[slot] << ','
          << g_standalone_grid[0] << ','
          << g_standalone_grid[1] << ',' << g_standalone_grid[2] << ','
          << g_standalone_block[0] << ',' << g_standalone_block[1] << ','
          << g_standalone_block[2] << ',' << cta << ',' << cta_x << ','
          << cta_y << ',' << cta_z << ','
          << (entry ? record.sm_id : 0) << ','
          << (exit ? record.exit_sm_id : 0) << ','
          << (entry && exit && record.sm_id != record.exit_sm_id) << ','
          << (entry ? record.nsmid : 0) << ','
          << (entry ? record.grid_id : 0) << ','
          << (entry ? record.entry_ns : 0) << ','
          << (exit ? record.exit_ns : 0) << ',' << entry_ns << ','
          << exit_ns << ','
          << (entry && exit ? record.exit_ns - record.entry_ns : 0) << ','
          << clock_error << ',' << entry << ',' << exit << ','
          << (entry && exit ? "complete" : entry ? "missing_exit" :
              exit ? "missing_entry" : "missing") << '\n';
    }
  }
  csv.close();
  std::rename(csv_temporary.c_str(), csv_path.c_str());

  const std::string status_path = g_standalone_output + "_cta_status.json";
  const std::string status_temporary = status_path + ".tmp";
  std::ofstream status(status_temporary);
  status << "{\n  \"schema\": \"pperf_nvbit_cta_standalone_v1\",\n"
         << "  \"mode\": \"controlled\",\n"
         << "  \"model_id\": " << json_string(g_standalone_model) << ",\n"
         << "  \"pid\": " << getpid() << ",\n"
         << "  \"tracker_version\": "
         << json_string(PPERF_NVBIT_CTA_VERSION) << ",\n"
         << "  \"kernel_name\": " << json_string(g_standalone_name) << ",\n"
         << "  \"tracker_error\": " << g_error.load() << ",\n"
         << "  \"launch_manipulation\": "
         << (g_aligned_labels.empty() ? "false" : "true") << ",\n"
         << "  \"matching_launch_count\": " << g_matching_occurrence
         << ",\n  \"clock_calibration\": {\n"
         << "    \"host_before_ns\": " << g_calibration_host_before_ns
         << ",\n    \"host_after_ns\": " << g_calibration_host_after_ns
         << ",\n    \"globaltimer_ns\": " << g_calibration_timer_ns
         << ",\n    \"offset_ns\": " << offset
         << ",\n    \"error_ns\": " << clock_error
         << "\n  },\n  \"targets\": [\n";
  for (std::size_t slot = 0; slot < g_launch_count; ++slot) {
    if (slot) status << ",\n";
    status << "    {\"label\": " << json_string(g_target_labels[slot])
           << ", \"launch_slot\": " << slot
           << ", \"target_occurrence\": " << g_target_occurrences[slot]
           << ", \"launch_observed\": "
           << (g_target_observed[slot] ? "true" : "false")
           << ", \"context_handle\": " << g_target_contexts[slot]
           << ", \"stream_handle\": " << g_target_streams[slot]
           << ", \"function_handle\": " << g_target_functions[slot]
           << ", \"launch_alignment_requested\": "
           << (contains(g_aligned_labels, g_target_labels[slot])
                   ? "true" : "false")
           << ", \"launch_alignment_complete\": "
           << (g_barrier_complete[slot] ? "true" : "false")
           << ", \"launch_alignment_wait_ns\": "
           << g_barrier_wait_ns[slot]
           << ", \"expected_count\": " << expected
           << ", \"entered_count\": " << entered[slot]
           << ", \"exited_count\": " << exited[slot]
           << ", \"complete_count\": " << complete[slot]
           << ", \"missing_count\": " << expected - complete[slot]
           << ", \"coverage_fraction\": "
           << (expected == 0 ? 0.0 : static_cast<double>(complete[slot]) /
                                  static_cast<double>(expected)) << '}';
  }
  status << "\n  ]\n}\n";
  status.close();
  std::rename(status_temporary.c_str(), status_path.c_str());
}

}  // namespace

namespace {

void observe_passive_launch(CUcontext context, nvbit_api_cuda_t cbid,
                            const LaunchView& launch) {
  const auto host_launch_ready_ns = monotonic_ns();
  if (g_error) return;
  const int target_index = g_passive_target.load();
  if (target_index < 0 ||
      static_cast<std::size_t>(target_index) >= g_target_labels.size()) {
    prepare_passive_function(context, launch.function);
    if (g_passive_learning && !g_error) {
      const auto sequence = ++g_passive_sequence;
      if (g_passive_sequence_functions.size() <= sequence) {
        g_passive_sequence_functions.resize(sequence + 1);
      }
      g_passive_sequence_functions[sequence].insert({context, launch.function});
    }
    return;
  }
  const auto target = static_cast<std::size_t>(target_index);
  const auto sequence = ++g_passive_sequence;
  g_passive_observed_sequences[target] = sequence;
  const char* observed_name = nvbit_get_func_name(context, launch.function);
  const std::string name = observed_name == nullptr ? "unknown" : observed_name;
  const std::string api = driver_launch_api(cbid);
  int registers_per_thread = -1;
  int static_shared_memory = -1;
  cuFuncGetAttribute(&registers_per_thread, CU_FUNC_ATTRIBUTE_NUM_REGS,
                     launch.function);
  cuFuncGetAttribute(&static_shared_memory,
                     CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, launch.function);
  const auto signature = launch_signature_key(
      name, api, launch.grid, launch.block, launch.dynamic_shared_memory,
      registers_per_thread, static_shared_memory);
  const auto occurrence = g_passive_occurrences[signature]++;
  if (g_mixed && sequence == g_passive_sequence_starts[target]) {
    if (!standalone_match(context, launch) ||
        !pperf::mixed_alignment_selected(
            sequence, g_passive_sequence_starts[target], occurrence,
            static_cast<std::uint32_t>(g_target_occurrences[target]))) {
      g_error = 43;
      return;
    }
    g_target_contexts[target] = reinterpret_cast<std::uint64_t>(context);
    g_target_streams[target] = reinterpret_cast<std::uint64_t>(launch.stream);
    g_target_functions[target] =
        reinterpret_cast<std::uint64_t>(launch.function);
    launch_barrier(target);
    if (g_error) return;
    g_target_observed[target] = true;
  }
  if (!pperf::passive_sequence_selected(
          sequence, g_passive_sequence_starts[target],
          g_passive_sequence_ends[target])) {
    refresh_passive_launch(context, launch.function,
                           pperf::kNvbitCtaInactiveLaunch);
    return;
  }

  const std::uint64_t expected =
      static_cast<std::uint64_t>(launch.grid[0]) * launch.grid[1] *
      launch.grid[2];
  g_expected_record_count += expected;
  PassiveLaunch metadata;
  metadata.target_label = g_target_labels[target];
  metadata.kernel_name = name;
  metadata.driver_launch_api = api;
  metadata.launch_slot = std::numeric_limits<std::uint32_t>::max();
  metadata.launch_sequence_index = sequence;
  metadata.occurrence_within_signature = occurrence;
  metadata.context_handle = reinterpret_cast<std::uint64_t>(context);
  metadata.stream_handle = reinterpret_cast<std::uint64_t>(launch.stream);
  metadata.function_handle =
      reinterpret_cast<std::uint64_t>(launch.function);
  metadata.host_launch_ready_ns = host_launch_ready_ns;
  std::copy(std::begin(launch.grid), std::end(launch.grid),
            std::begin(metadata.grid));
  std::copy(std::begin(launch.block), std::end(launch.block),
            std::begin(metadata.block));
  metadata.dynamic_shared_memory = launch.dynamic_shared_memory;
  metadata.expected_count = expected;
  metadata.registers_per_thread = registers_per_thread;
  metadata.static_shared_memory = static_shared_memory;
  const bool capacity_available =
      expected <= g_record_capacity - g_record_count;
  if (g_launch_count >= pperf::kNvbitCtaMaxLaunches ||
      !capacity_available) {
    ++g_dropped_launch_count;
    g_dropped_capacity_record_count += expected;
    g_passive_launches.push_back(std::move(metadata));
    refresh_passive_launch(context, launch.function,
                           pperf::kNvbitCtaInactiveLaunch);
    return;
  }

  const auto slot = static_cast<std::uint32_t>(g_launch_count++);
  metadata.launch_slot = slot;
  metadata.storage_reserved = true;
  g_launches[slot] = {
      reinterpret_cast<std::uint64_t>(launch.function), slot,
      launch.grid[0], launch.grid[1], launch.grid[2],
      launch.block[0], launch.block[1], launch.block[2]};
  g_bases[slot] = static_cast<std::uint32_t>(g_record_count);
  g_threads[slot] = launch.block[0] * launch.block[1] * launch.block[2];
  g_record_count += static_cast<std::size_t>(expected);
  if (!select_passive_launch(context, launch.function, slot)) {
    g_error = 35;
    g_passive_launches.push_back(std::move(metadata));
    return;
  }
  g_target_observed[target] = true;
  g_passive_launches.push_back(std::move(metadata));
}

}  // namespace

void nvbit_at_cuda_event(CUcontext context, int is_exit,
                         nvbit_api_cuda_t cbid, const char*, void* params,
                         CUresult*) {
  if (!launch_event(cbid) || is_exit) return;
  const auto launch = launch_view(cbid, params);
  if (g_standalone) {
    if (g_passive) {
      observe_passive_launch(context, cbid, launch);
      return;
    }
    if (g_error || !standalone_match(context, launch)) return;
    ++g_matching_occurrence;
    const int armed = g_armed_slot.load();
    if (armed < 0) {
      if (g_instrumented.count({context, launch.function})) {
        nvbit_enable_instrumented(context, launch.function, false, false);
      }
      return;
    }
    const auto slot = static_cast<std::size_t>(armed);
    if (g_within_input_occurrence++ != g_target_occurrences[slot]) return;
    g_launches[slot].function_handle =
        reinterpret_cast<std::uint64_t>(launch.function);
    g_target_contexts[slot] = reinterpret_cast<std::uint64_t>(context);
    g_target_streams[slot] = reinterpret_cast<std::uint64_t>(launch.stream);
    g_target_functions[slot] =
        reinterpret_cast<std::uint64_t>(launch.function);
    if (!instrument(context, launch.function)) {
      g_error = 6;
      return;
    }
    launch_barrier(slot);
    if (g_error) return;
    nvbit_set_at_launch(context, launch.function, slot);
    nvbit_enable_instrumented(context, launch.function, true, false);
    g_target_observed[slot] = true;
    g_armed_slot = -1;
    return;
  }
  const int slot = g_active ? g_next_slot.exchange(-1) : -1;
  if (slot < 0 || static_cast<std::size_t>(slot) >= g_launch_count) {
    nvbit_enable_instrumented(context, launch.function, false, false);
    return;
  }
  const auto& expected = g_launches[slot];
  if (expected.launch_slot != static_cast<std::uint32_t>(slot) ||
      !matches(launch, expected)) {
    g_error = 7;
    nvbit_enable_instrumented(context, launch.function, false, false);
    return;
  }
  if (!instrument(context, launch.function)) {
    g_error = 6;
    nvbit_enable_instrumented(context, launch.function, false, false);
    return;
  }
  nvbit_set_at_launch(context, launch.function,
                      static_cast<std::uint64_t>(slot));
  nvbit_enable_instrumented(context, launch.function, true, false);
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_configure_from_environment() {
  if (!g_standalone) nvbit_at_init();
  return g_standalone && !g_error ? 0 : 42;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_input_begin() {
  if (!g_standalone) return 36;
  if (!g_passive) return 38;
  if (g_error) return g_error.load();
  if (g_passive_target.load() >= 0) return 39;
  if (g_passive_learning.exchange(true)) return 40;
  g_passive_sequence = 0;
  return 0;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_input_end() {
  if (!g_standalone || !g_passive) return 37;
  if (g_error) return g_error.load();
  if (!g_passive_learning.exchange(false)) return 41;
  ++g_passive_learned_inputs;
  return 0;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_arm(const char* label) {
  if (!g_standalone || label == nullptr || g_error) return 30;
  const auto found = std::find(
      g_target_labels.begin(), g_target_labels.end(), std::string(label));
  if (found == g_target_labels.end()) return 31;
  const auto slot = static_cast<int>(
      std::distance(g_target_labels.begin(), found));
  if (g_passive) {
    if (g_passive_learning || !g_enabled.empty() ||
        !g_passive_predicted_functions.empty()) return 34;
    g_passive_occurrences.clear();
    g_passive_sequence = 0;
    const auto start = g_passive_sequence_starts[slot];
    const auto end = std::min<std::size_t>(
        g_passive_sequence_ends[slot],
        g_passive_sequence_functions.empty()
            ? 0 : g_passive_sequence_functions.size() - 1);
    for (std::size_t sequence = start; sequence <= end; ++sequence) {
      for (const auto& function : g_passive_sequence_functions[sequence]) {
        g_passive_predicted_functions.insert(function);
      }
    }
    g_passive_prediction_fallbacks[slot] =
        g_passive_predicted_functions.empty();
    g_passive_predicted_function_counts[slot] =
        g_passive_predicted_functions.size();
    std::unordered_set<InstrumentedFunction, InstrumentedFunctionHash>
        seen_before;
    for (std::size_t sequence = 1;
         sequence < std::min<std::size_t>(
             start, g_passive_sequence_functions.size());
         ++sequence) {
      for (const auto& function : g_passive_sequence_functions[sequence]) {
        seen_before.insert(function);
      }
    }
    g_passive_prearmed_slots.clear();
    for (std::size_t sequence = start; sequence <= end; ++sequence) {
      for (const auto& function : g_passive_sequence_functions[sequence]) {
        if (seen_before.count(function)) continue;
        const auto launch_slot = static_cast<std::uint32_t>(sequence - start);
        const auto [found, inserted] =
            g_passive_prearmed_slots.emplace(function, launch_slot);
        if (!inserted && found->second != launch_slot) {
          g_passive_prearmed_slots.erase(found);
          seen_before.insert(function);
        }
      }
    }
    for (const auto& [function, launch_slot] : g_passive_prearmed_slots) {
      if (!g_instrumented.count(function)) continue;
      nvbit_set_at_launch(
          function.context, function.function, launch_slot);
      if (g_enabled.insert(function).second) {
        nvbit_enable_instrumented(
            function.context, function.function, true, false);
        ++g_passive_enable_transitions;
      }
    }
    g_passive_target = slot;
  } else {
    g_within_input_occurrence = 0;
    g_armed_slot = slot;
  }
  return 0;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_disarm() {
  if (!g_standalone || g_error) return 33;
  if (!g_passive) return 0;
  const int target = g_passive_target.exchange(-1);
  if (target >= 0 &&
      static_cast<std::size_t>(target) < g_passive_disarmed.size()) {
    g_passive_disarmed[target] = true;
    g_passive_observed_sequences[target] = g_passive_sequence;
  }
  for (const auto& function : g_enabled) {
    nvbit_enable_instrumented(
        function.context, function.function, false, false);
    ++g_passive_disable_transitions;
  }
  g_enabled.clear();
  g_passive_predicted_functions.clear();
  g_passive_prearmed_slots.clear();
  return 0;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_probe(std::uint32_t abi_version,
                          const char** tracker_version) {
  if (abi_version != PPERF_NVBIT_CTA_ABI_VERSION ||
      tracker_version == nullptr) return 1;
  *tracker_version = PPERF_NVBIT_CTA_VERSION;
  return 0;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_prepare(const pperf_nvbit_cta_launch_spec* launches,
                            std::size_t launch_count) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_active = false;
  g_next_slot = -1;
  g_error = 0;
  g_launch_count = 0;
  g_record_count = 0;
  g_expected_record_count = 0;
  g_dropped_capacity_record_count = 0;
  g_dropped_launch_count = 0;
  if (!initialize_storage()) return 12;
  if (g_records == nullptr || g_bases == nullptr || g_threads == nullptr ||
      g_calibration_globaltimer == nullptr) return 12;
  if (launches == nullptr || launch_count == 0 ||
      launch_count > pperf::kNvbitCtaMaxLaunches) return 2;
  std::uint64_t total = 0;
  for (std::size_t index = 0; index < launch_count; ++index) {
    const auto& launch = launches[index];
    if (launch.launch_slot != index || launch.grid_x == 0 ||
        launch.grid_y == 0 || launch.grid_z == 0 || launch.block_x == 0 ||
        launch.block_y == 0 || launch.block_z == 0) return 3;
    g_launches[index] = launch;
    g_bases[index] = static_cast<std::uint32_t>(total);
    g_threads[index] = launch.block_x * launch.block_y * launch.block_z;
    total += static_cast<std::uint64_t>(launch.grid_x) * launch.grid_y *
             launch.grid_z;
    if (total > g_record_capacity) return 4;
  }
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr) {
    return 13;
  }
  for (std::size_t index = 0; index < launch_count; ++index) {
    auto function = reinterpret_cast<CUfunction>(
        g_launches[index].function_handle);
    if (!instrument(context, function)) {
      g_error = 6;
      return g_error.load();
    }
    nvbit_enable_instrumented(context, function, false, false);
  }
  g_launch_count = launch_count;
  g_record_count = static_cast<std::size_t>(total);
  g_expected_record_count = total;
  for (std::size_t index = 0; index < g_record_count; ++index) {
    g_records[index] = {
        std::numeric_limits<unsigned long long>::max(), 0,
        std::numeric_limits<unsigned long long>::max(),
        std::numeric_limits<unsigned int>::max(),
        std::numeric_limits<unsigned int>::max(), 0,
        std::numeric_limits<unsigned int>::max()};
  }
  g_active = true;
  return 0;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_identify_next_launch(std::uint32_t launch_slot) {
  if (!g_active || launch_slot >= g_launch_count || g_error) return 5;
  g_next_slot = static_cast<int>(launch_slot);
  return 0;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_calibrate(std::uint64_t* globaltimer_ns) {
  if (globaltimer_ns == nullptr) return 8;
  g_next_slot = -1;
  calibrate_globaltimer<<<1, 1>>>(g_calibration_globaltimer);
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) return 9;
  *globaltimer_ns = *g_calibration_globaltimer;
  return 0;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_collect(pperf_nvbit_cta_record* records,
                            std::size_t capacity,
                            std::size_t* record_count) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_active = false;
  g_next_slot = -1;
  if (record_count == nullptr || capacity < g_record_count ||
      (records == nullptr && g_record_count != 0)) return 10;
  if (g_error) return g_error.load();
  std::size_t output = 0;
  for (std::size_t slot = 0; slot < g_launch_count; ++slot) {
    const auto expected = static_cast<std::size_t>(g_launches[slot].grid_x) *
                          g_launches[slot].grid_y *
                          g_launches[slot].grid_z;
    for (std::size_t cta = 0; cta < expected; ++cta) {
      const auto& source = g_records[g_bases[slot] + cta];
      auto& target = records[output++];
      target.launch_slot = static_cast<std::uint32_t>(slot);
      target.cta_id = static_cast<std::uint32_t>(cta);
      target.sm_id = source.sm_id;
      target.exit_sm_id = source.exit_sm_id;
      target.nsmid = source.nsmid;
      target.grid_id = source.grid_id;
      target.observation_flags =
          (source.entry_ns != std::numeric_limits<unsigned long long>::max()
               ? PPERF_NVBIT_CTA_ENTRY_OBSERVED
               : 0U) |
          (source.exit_ns != 0 ? PPERF_NVBIT_CTA_EXIT_OBSERVED : 0U);
      target.entry_globaltimer_ns =
          target.observation_flags & PPERF_NVBIT_CTA_ENTRY_OBSERVED
              ? source.entry_ns
              : 0;
      target.exit_globaltimer_ns = source.exit_ns;
    }
  }
  *record_count = output;
  return 0;
}

extern "C" __attribute__((visibility("default")))
int pperf_nvbit_cta_get_collection_status(
    pperf_nvbit_cta_collection_status* status) {
  if (status == nullptr) return 11;
  status->record_capacity = g_record_capacity;
  status->record_high_water_mark = g_record_count;
  status->expected_record_count = g_expected_record_count;
  status->reserved_record_count = g_record_count;
  status->dropped_capacity_record_count =
      g_dropped_capacity_record_count;
  status->incomplete_entry_count = 0;
  status->incomplete_exit_count = 0;
  for (std::size_t index = 0; index < g_record_count; ++index) {
    status->incomplete_entry_count +=
        g_records[index].entry_ns ==
        std::numeric_limits<unsigned long long>::max();
    status->incomplete_exit_count += g_records[index].exit_ns == 0;
  }
  return 0;
}
