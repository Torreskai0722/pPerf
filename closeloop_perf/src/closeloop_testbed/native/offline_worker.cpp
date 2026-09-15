#include "offline_workload_abi.h"
#include "nvbit_cta_abi.h"
#include "nvbit_cta_tracker_logic.h"

#include <cupti.h>
#include <cupti_activity.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <time.h>

using json = nlohmann::json;

namespace {

struct Timing {
  std::uint64_t start{};
  std::uint64_t end{};
};

std::atomic<bool> g_timing_active{false};
std::vector<std::uint32_t> g_correlations;
std::map<std::uint32_t, Timing> g_activity;
void* g_agent_library{};

void* agent_symbol(const char* name) {
  void* value = dlsym(RTLD_DEFAULT, name);
  if (value != nullptr) return value;
  const char* path = std::getenv("CUDA_INJECTION64_PATH");
  if (path == nullptr || *path == '\0') return nullptr;
  if (g_agent_library == nullptr) {
    g_agent_library = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
    if (g_agent_library == nullptr) {
      g_agent_library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    }
  }
  return g_agent_library == nullptr ? nullptr : dlsym(g_agent_library, name);
}

void require_cuda(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* error = nullptr;
  cuGetErrorString(result, &error);
  throw std::runtime_error(std::string(operation) + ": " +
                           (error == nullptr ? std::to_string(result) : error));
}

void require_cupti(CUptiResult result, const char* operation) {
  if (result == CUPTI_SUCCESS) return;
  const char* error = nullptr;
  cuptiGetResultString(result, &error);
  throw std::runtime_error(std::string(operation) + ": " +
                           (error == nullptr ? std::to_string(result) : error));
}

std::string hex(const unsigned char* value, std::size_t size) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < size; ++index) {
    output << std::setw(2) << static_cast<unsigned int>(value[index]);
  }
  return output.str();
}

std::string sha256(const void* value, std::size_t size) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char*>(value), size, digest);
  return hex(digest, sizeof(digest));
}

std::uint64_t monotonic_ns() {
  timespec value{};
  clock_gettime(CLOCK_MONOTONIC, &value);
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

void sleep_until(std::uint64_t release_ns) {
  timespec value{};
  value.tv_sec = static_cast<time_t>(release_ns / 1000000000ULL);
  value.tv_nsec = static_cast<long>(release_ns % 1000000000ULL);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &value, nullptr) ==
         EINTR) {}
}

void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                       const void* raw) {
  if (!g_timing_active || domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
  if (cbid != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel &&
      cbid != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) return;
  const auto* data = static_cast<const CUpti_CallbackData*>(raw);
  if (data->callbackSite == CUPTI_API_ENTER) {
    g_correlations.push_back(data->correlationId);
  }
}

void CUPTIAPI buffer_requested(std::uint8_t** buffer, std::size_t* size,
                               std::size_t* max_records) {
  constexpr std::size_t kSize = 1024 * 1024;
  void* storage = nullptr;
  if (posix_memalign(&storage, 8, kSize) != 0) storage = nullptr;
  *buffer = static_cast<std::uint8_t*>(storage);
  *size = storage == nullptr ? 0 : kSize;
  *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, std::uint32_t, std::uint8_t* buffer,
                               std::size_t, std::size_t valid_size) {
  CUpti_Activity* record = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid_size, &record) ==
         CUPTI_SUCCESS) {
    if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL &&
        record->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
    const auto* kernel =
        reinterpret_cast<const CUpti_ActivityKernel9*>(record);
    g_activity[kernel->correlationId] = {kernel->start, kernel->end};
  }
  std::free(buffer);
}

std::string argument(int argc, char** argv, const std::string& name,
                     const std::string& fallback = "") {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) return argv[index + 1];
  }
  return fallback;
}

bool flag(int argc, char** argv, const std::string& name) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] == name) return true;
  }
  return false;
}

class Worker {
 public:
  Worker(const std::string& adapter_path, const std::string& configuration,
         const std::string& client_id, bool timing, bool cta)
      : adapter_path_(adapter_path), configuration_(configuration),
        client_id_(client_id), timing_(timing), cta_(cta) {
    require_cuda(cuInit(0), "cuInit");
    CUdevice device{};
    require_cuda(cuDeviceGet(&device, 0), "cuDeviceGet");
    require_cuda(cuCtxCreate(&context_, 0, device), "cuCtxCreate");
    adapter_library_ = dlopen(adapter_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (adapter_library_ == nullptr) throw std::runtime_error(dlerror());
    auto entry = reinterpret_cast<pperf_offline_workload_v1_entry_fn>(
        dlsym(adapter_library_, "pperf_offline_workload_v1_entry"));
    if (entry == nullptr) throw std::runtime_error("adapter entry is absent");
    api_ = entry();
    if (api_ == nullptr ||
        api_->abi_version != PPERF_OFFLINE_WORKLOAD_ABI_VERSION ||
        api_->module_image == nullptr || api_->create == nullptr ||
        api_->reset == nullptr || api_->launch_count == nullptr ||
        api_->launch == nullptr || api_->output_count == nullptr ||
        api_->output == nullptr || api_->destroy == nullptr) {
      throw std::runtime_error("adapter ABI/version mismatch");
    }
    const void* image = nullptr;
    std::size_t image_size = 0;
    if (api_->module_image(&image, &image_size) != 0 || image == nullptr ||
        image_size == 0) throw std::runtime_error("adapter module is invalid");
    module_sha256_ = sha256(image, image_size);
    char jit_log[4096]{};
    CUjit_option options[] = {
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* values[] = {
        jit_log, reinterpret_cast<void*>(sizeof(jit_log))};
    const CUresult module_result = cuModuleLoadDataEx(
        &module_, image, 2, options, values);
    if (module_result != CUDA_SUCCESS) {
      throw std::runtime_error(std::string("cuModuleLoadDataEx: ") + jit_log);
    }
    char error[512]{};
    if (api_->create(configuration.c_str(), module_, &state_, error,
                     sizeof(error)) != 0 || state_ == nullptr) {
      throw std::runtime_error(std::string("adapter create failed: ") + error);
    }
    require_cuda(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
                 "cuStreamCreate");
    snapshot_descriptors();
    register_agent_outputs();
    if (timing_) start_timing();
    if (cta_) load_cta();
  }

  ~Worker() {
    if (subscriber_ != nullptr) cuptiUnsubscribe(subscriber_);
    if (timing_) cuptiActivityDisable(
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (stream_ != nullptr) cuStreamDestroy(stream_);
    if (api_ != nullptr && state_ != nullptr) api_->destroy(state_);
    if (module_ != nullptr) cuModuleUnload(module_);
    if (context_ != nullptr) cuCtxDestroy(context_);
    if (adapter_library_ != nullptr) dlclose(adapter_library_);
  }

  json identity() const {
    return {{"client_id", client_id_},
            {"adapter_name", api_->adapter_name},
            {"module_sha256", module_sha256_},
            {"configuration", json::parse(configuration_)},
            {"launches", descriptor_identity_},
            {"workload_fingerprint", workload_fingerprint_}};
  }

  void warmup() {
    reset();
    for (const auto& launch : descriptors_) issue(launch);
    require_cuda(cuStreamSynchronize(stream_), "warmup synchronize");
  }

  void prepare_capture() { reset(); }

  void capture_launch() {
    validate_descriptors();
    for (const auto& launch : descriptors_) issue(launch);
    require_cuda(cuStreamSynchronize(stream_), "capture synchronize");
  }

  json run(std::uint64_t release_ns, bool enabled) {
    reset();
    g_correlations.clear();
    g_activity.clear();
    sleep_until(release_ns);
    std::uint64_t replay_epoch = 0;
    require_cupti(cuptiGetTimestamp(&replay_epoch), "cupti replay epoch");
    json activities = json::array();
    std::vector<std::uint64_t> issue_starts;
    std::vector<std::uint64_t> issue_ends;
    if (enabled) {
      validate_descriptors();
      g_timing_active = true;
      for (const auto& launch : descriptors_) {
        std::uint64_t start = 0;
        std::uint64_t end = 0;
        require_cupti(cuptiGetTimestamp(&start), "cupti issue start");
        issue(launch);
        require_cupti(cuptiGetTimestamp(&end), "cupti issue end");
        issue_starts.push_back(start);
        issue_ends.push_back(end);
      }
      g_timing_active = false;
      require_cuda(cuStreamSynchronize(stream_), "timed synchronize");
      require_cupti(cuptiActivityFlushAll(
                        CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
                    "cupti activity flush");
      if (g_correlations.size() != descriptors_.size()) {
        throw std::runtime_error("hidden or missing timed launch");
      }
      for (std::size_t index = 0; index < descriptors_.size(); ++index) {
        const auto found = g_activity.find(g_correlations[index]);
        if (found == g_activity.end() || found->second.end <= found->second.start) {
          throw std::runtime_error("CUPTI activity is incomplete");
        }
        const auto& launch = descriptors_[index];
        activities.push_back({
            {"ordinal", index}, {"symbol", launch.symbol},
            {"driver_issue_start_ns", issue_starts[index]},
            {"driver_issue_end_ns", issue_ends[index]},
            {"gpu_start_ns", found->second.start},
            {"gpu_end_ns", found->second.end},
            {"gpu_duration_ns", found->second.end - found->second.start},
            {"issue_to_gpu_start_wait_ns",
             static_cast<std::int64_t>(found->second.start) -
                 static_cast<std::int64_t>(issue_ends[index])},
            {"replay_ready_ns", issue_ends[index]},
            {"admission_wait_ns",
             static_cast<std::int64_t>(found->second.start) -
                 static_cast<std::int64_t>(issue_ends[index])},
            {"grid", launch_json(launch)["grid"]},
            {"block", launch_json(launch)["block"]},
            {"dynamic_shared_memory", launch.dynamic_shared_memory}});
      }
    }
    return {{"client_id", client_id_}, {"enabled", enabled},
            {"release_ns", release_ns}, {"replay_epoch_ns", replay_epoch},
            {"launch_activities", activities},
            {"launch_count", activities.size()},
            {"output_sha256", digest_outputs()},
            {"workload_fingerprint", workload_fingerprint_}};
  }

  json cta_run(std::uint64_t release_ns) {
    reset();
    std::vector<pperf_nvbit_cta_launch_spec> launches;
    std::size_t expected = 0;
    for (std::size_t index = 0; index < descriptors_.size(); ++index) {
      const auto& value = descriptors_[index];
      launches.push_back({reinterpret_cast<std::uint64_t>(value.function),
                          static_cast<std::uint32_t>(index), value.grid[0],
                          value.grid[1], value.grid[2], value.block[0],
                          value.block[1], value.block[2]});
      expected += value.grid[0] * value.grid[1] * value.grid[2];
    }
    const int prepared = cta_prepare_(launches.data(), launches.size());
    if (prepared != 0) {
      throw std::runtime_error(
          "NVBit CTA prepare failed: " + std::to_string(prepared));
    }
    std::uint64_t before = 0;
    std::uint64_t after = 0;
    std::uint64_t globaltimer = 0;
    require_cupti(cuptiGetTimestamp(&before), "CTA calibration start");
    if (cta_calibrate_(&globaltimer) != 0) {
      throw std::runtime_error("NVBit CTA calibration failed");
    }
    require_cupti(cuptiGetTimestamp(&after), "CTA calibration end");
    const std::uint64_t midpoint = before + (after - before) / 2;
    const std::int64_t offset = midpoint >= globaltimer
        ? static_cast<std::int64_t>(midpoint - globaltimer)
        : -static_cast<std::int64_t>(globaltimer - midpoint);
    sleep_until(release_ns);
    for (std::size_t index = 0; index < descriptors_.size(); ++index) {
      const int identified = cta_identify_(index);
      if (identified != 0) {
        throw std::runtime_error(
            "NVBit CTA launch identification failed: " +
            std::to_string(identified));
      }
      issue(descriptors_[index]);
    }
    require_cuda(cuStreamSynchronize(stream_), "CTA synchronize");
    std::vector<pperf_nvbit_cta_record> records(expected);
    std::size_t observed = 0;
    const int collected = cta_collect_(
        records.data(), records.size(), &observed);
    if (collected != 0 || observed != expected) {
      throw std::runtime_error(
          "NVBit CTA collection failed: " + std::to_string(collected));
    }
    pperf_nvbit_cta_collection_status collection{};
    if (cta_status_(&collection) != 0) {
      throw std::runtime_error("NVBit CTA status collection failed");
    }
    json intervals = json::array();
    std::vector<std::size_t> entered(descriptors_.size());
    std::vector<std::size_t> exited(descriptors_.size());
    std::vector<std::size_t> complete(descriptors_.size());
    for (const auto& record : records) {
      const bool has_entry = record.observation_flags &
          PPERF_NVBIT_CTA_ENTRY_OBSERVED;
      const bool has_exit = record.observation_flags &
          PPERF_NVBIT_CTA_EXIT_OBSERVED;
      std::uint64_t entry = 0;
      std::uint64_t exit = 0;
      if (has_entry && !pperf::calibrated_timestamp(
              record.entry_globaltimer_ns, offset, entry)) {
        throw std::runtime_error("invalid CTA entry calibration");
      }
      if (has_exit && !pperf::calibrated_timestamp(
              record.exit_globaltimer_ns, offset, exit)) {
        throw std::runtime_error("invalid CTA exit calibration");
      }
      entered[record.launch_slot] += has_entry;
      exited[record.launch_slot] += has_exit;
      complete[record.launch_slot] += has_entry && has_exit;
      intervals.push_back({
          {"ordinal", record.launch_slot}, {"cta_id", record.cta_id},
          {"sm_id", record.sm_id == std::numeric_limits<std::uint32_t>::max()
              ? json(nullptr) : json(record.sm_id)},
          {"exit_sm_id",
           record.exit_sm_id == std::numeric_limits<std::uint32_t>::max()
               ? json(nullptr) : json(record.exit_sm_id)},
          {"sm_migrated",
           record.sm_id != std::numeric_limits<std::uint32_t>::max() &&
               record.exit_sm_id != std::numeric_limits<std::uint32_t>::max() &&
               record.sm_id != record.exit_sm_id},
          {"nsmid", record.nsmid},
          {"grid_id", record.grid_id},
          {"entry_globaltimer_ns",
           has_entry ? json(record.entry_globaltimer_ns) : json(nullptr)},
          {"exit_globaltimer_ns",
           has_exit ? json(record.exit_globaltimer_ns) : json(nullptr)},
          {"entry_ns", has_entry ? json(entry) : json(nullptr)},
          {"exit_ns", has_exit ? json(exit) : json(nullptr)},
          {"clock_error_ns", (after - before + 1) / 2},
          {"entry_observed", has_entry}, {"exit_observed", has_exit},
          {"observation_status", has_entry && has_exit ? "complete" :
              has_entry ? "entry_only" : has_exit ? "exit_only" : "missing"}});
    }
    json statuses = json::array();
    for (std::size_t index = 0; index < descriptors_.size(); ++index) {
      const std::size_t count = descriptors_[index].grid[0] *
          descriptors_[index].grid[1] * descriptors_[index].grid[2];
      statuses.push_back({
          {"ordinal", index}, {"expected_count", count},
          {"entered_count", entered[index]}, {"exited_count", exited[index]},
          {"complete_count", complete[index]},
          {"missing_count", count - complete[index]},
          {"dropped_entry_count", count - entered[index]},
          {"dropped_exit_count", count - exited[index]},
          {"coverage_fraction", count == 0 ? 0.0 :
              static_cast<double>(complete[index]) / count},
          {"quality", complete[index] == count ? "complete" : "partial"}});
    }
    return {{"client_id", client_id_}, {"cta_intervals", intervals},
            {"cta_collection_status", statuses},
            {"cta_buffer_status", {
                {"record_capacity", collection.record_capacity},
                {"record_high_water_mark",
                 collection.record_high_water_mark},
                {"expected_record_count",
                 collection.expected_record_count},
                {"reserved_record_count",
                 collection.reserved_record_count},
                {"dropped_capacity_record_count",
                 collection.dropped_capacity_record_count},
                {"incomplete_entry_count",
                 collection.incomplete_entry_count},
                {"incomplete_exit_count",
                 collection.incomplete_exit_count}}},
            {"clock_calibrations", json::array({{
                {"client_id", client_id_}, {"clock_a", "globaltimer"},
                {"clock_b", "cupti_timestamp"}, {"offset_ns", offset},
                {"error_ns", (after - before + 1) / 2},
                {"precision_class", "bounded_profiler_calibration"}}})}};
  }

 private:
  static json launch_json(const pperf_offline_launch_v1& value) {
    return {{"symbol", value.symbol == nullptr ? "" : value.symbol},
            {"launch_api", "cuLaunchKernel"},
            {"grid", {value.grid[0], value.grid[1], value.grid[2]}},
            {"block", {value.block[0], value.block[1], value.block[2]}},
            {"dynamic_shared_memory", value.dynamic_shared_memory}};
  }

  void snapshot_descriptors() {
    const std::size_t count = api_->launch_count(state_);
    if (count == 0 || count > 256) {
      throw std::runtime_error("adapter launch count is invalid");
    }
    for (std::size_t index = 0; index < count; ++index) {
      pperf_offline_launch_v1 launch{};
      if (api_->launch(state_, index, &launch) != 0 ||
          launch.function == nullptr || launch.symbol == nullptr ||
          launch.kernel_parameters == nullptr ||
          std::any_of(std::begin(launch.grid), std::end(launch.grid),
                      [](std::uint32_t value) { return value == 0; }) ||
          std::any_of(std::begin(launch.block), std::end(launch.block),
                      [](std::uint32_t value) { return value == 0; })) {
        throw std::runtime_error("adapter launch descriptor is malformed");
      }
      descriptors_.push_back(launch);
      json identity = launch_json(launch);
      int registers = 0;
      int static_shared = 0;
      int occupancy = 0;
      require_cuda(cuFuncGetAttribute(
          &registers, CU_FUNC_ATTRIBUTE_NUM_REGS, launch.function),
          "cuFuncGetAttribute registers");
      require_cuda(cuFuncGetAttribute(
          &static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
          launch.function), "cuFuncGetAttribute shared");
      require_cuda(cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, launch.function,
          launch.block[0] * launch.block[1] * launch.block[2],
          launch.dynamic_shared_memory), "cuOccupancyMaxActiveBlocks");
      identity.update({
          {"registers_per_thread", registers},
          {"static_shared_memory", static_shared},
          {"threads_per_block",
           launch.block[0] * launch.block[1] * launch.block[2]},
          {"warps_per_block",
           (launch.block[0] * launch.block[1] * launch.block[2] + 31) / 32},
          {"occupancy_blocks_per_sm", occupancy}});
      descriptor_identity_.push_back(identity);
    }
    const std::string encoded = json({{"adapter", api_->adapter_name},
        {"module_sha256", module_sha256_},
        {"configuration", json::parse(configuration_)},
        {"launches", descriptor_identity_}}).dump();
    workload_fingerprint_ = sha256(encoded.data(), encoded.size());
  }

  void validate_descriptors() {
    if (api_->launch_count(state_) != descriptors_.size()) {
      throw std::runtime_error("adapter launch count changed");
    }
    for (std::size_t index = 0; index < descriptors_.size(); ++index) {
      pperf_offline_launch_v1 value{};
      if (api_->launch(state_, index, &value) != 0 ||
          launch_json(value) != json({
              {"symbol", descriptor_identity_[index]["symbol"]},
              {"launch_api", descriptor_identity_[index]["launch_api"]},
              {"grid", descriptor_identity_[index]["grid"]},
              {"block", descriptor_identity_[index]["block"]},
              {"dynamic_shared_memory",
               descriptor_identity_[index]["dynamic_shared_memory"]}}) ||
          value.function != descriptors_[index].function ||
          value.kernel_parameters != descriptors_[index].kernel_parameters) {
        throw std::runtime_error("adapter launch descriptor changed");
      }
    }
  }

  void issue(const pperf_offline_launch_v1& launch) {
    require_cuda(cuLaunchKernel(
        launch.function, launch.grid[0], launch.grid[1], launch.grid[2],
        launch.block[0], launch.block[1], launch.block[2],
        launch.dynamic_shared_memory, stream_, launch.kernel_parameters,
        nullptr), "cuLaunchKernel");
  }

  void reset() {
    if (api_->reset(state_, stream_) != 0) {
      throw std::runtime_error("adapter reset failed");
    }
    require_cuda(cuStreamSynchronize(stream_), "reset synchronize");
  }

  std::string digest_outputs() {
    SHA256_CTX digest;
    SHA256_Init(&digest);
    const std::size_t count = api_->output_count(state_);
    for (std::size_t index = 0; index < count; ++index) {
      pperf_offline_output_v1 output{};
      if (api_->output(state_, index, &output) != 0 || output.pointer == 0 ||
          output.size == 0 || output.name == nullptr) {
        throw std::runtime_error("adapter output descriptor is malformed");
      }
      std::vector<unsigned char> bytes(output.size);
      require_cuda(cuMemcpyDtoH(bytes.data(), output.pointer, output.size),
                   "output copy");
      SHA256_Update(&digest, output.name, std::strlen(output.name));
      SHA256_Update(&digest, bytes.data(), bytes.size());
    }
    unsigned char result[SHA256_DIGEST_LENGTH];
    SHA256_Final(result, &digest);
    return hex(result, sizeof(result));
  }

  void start_timing() {
    require_cupti(cuptiActivityRegisterCallbacks(
                      buffer_requested, buffer_completed),
                  "cuptiActivityRegisterCallbacks");
    require_cupti(cuptiActivityEnable(
                      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
                  "cuptiActivityEnable");
    require_cupti(cuptiSubscribe(&subscriber_, callback, nullptr),
                  "cuptiSubscribe");
    require_cupti(cuptiEnableDomain(
                      1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API),
                  "cuptiEnableDomain");
  }

  void load_cta() {
    const char* version = nullptr;
    auto probe = reinterpret_cast<pperf_nvbit_cta_probe_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_probe"));
    cta_prepare_ = reinterpret_cast<pperf_nvbit_cta_prepare_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_prepare"));
    cta_identify_ = reinterpret_cast<pperf_nvbit_cta_identify_next_launch_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_identify_next_launch"));
    cta_calibrate_ = reinterpret_cast<pperf_nvbit_cta_calibrate_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_calibrate"));
    cta_collect_ = reinterpret_cast<pperf_nvbit_cta_collect_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_collect"));
    cta_status_ = reinterpret_cast<pperf_nvbit_cta_collection_status_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_get_collection_status"));
    if (probe == nullptr || cta_prepare_ == nullptr ||
        cta_identify_ == nullptr || cta_calibrate_ == nullptr ||
        cta_collect_ == nullptr || cta_status_ == nullptr ||
        probe(PPERF_NVBIT_CTA_ABI_VERSION, &version) != 0) {
      throw std::runtime_error("NVBit CTA tracker ABI is unavailable");
    }
  }

  void register_agent_outputs() {
    using Register = int (*)(CUdeviceptr, std::size_t, const char*);
    auto register_output = reinterpret_cast<Register>(
        agent_symbol("pperf_agent_register_output"));
    if (register_output == nullptr) return;
    for (std::size_t index = 0; index < api_->output_count(state_); ++index) {
      pperf_offline_output_v1 output{};
      if (api_->output(state_, index, &output) != 0 ||
          register_output(output.pointer, output.size, output.name) != 0) {
        throw std::runtime_error("live agent output registration failed");
      }
    }
  }

  std::string adapter_path_;
  std::string configuration_;
  std::string client_id_;
  bool timing_{};
  bool cta_{};
  void* adapter_library_{};
  const pperf_offline_workload_v1* api_{};
  void* state_{};
  CUcontext context_{};
  CUmodule module_{};
  CUstream stream_{};
  CUpti_SubscriberHandle subscriber_{};
  std::vector<pperf_offline_launch_v1> descriptors_;
  json descriptor_identity_ = json::array();
  std::string module_sha256_;
  std::string workload_fingerprint_;
  pperf_nvbit_cta_prepare_fn cta_prepare_{};
  pperf_nvbit_cta_identify_next_launch_fn cta_identify_{};
  pperf_nvbit_cta_calibrate_fn cta_calibrate_{};
  pperf_nvbit_cta_collect_fn cta_collect_{};
  pperf_nvbit_cta_collection_status_fn cta_status_{};
};

void write_status(const std::string& path, const json& value) {
  if (path.empty()) return;
  std::ofstream output(path);
  output << value.dump(2) << '\n';
}

template <typename Function>
Function required_symbol(const char* name) {
  auto value = reinterpret_cast<Function>(agent_symbol(name));
  if (value == nullptr) throw std::runtime_error(std::string(name) + " absent");
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string adapter = argument(argc, argv, "--adapter");
    const std::string configuration = argument(argc, argv, "--config-json");
    const std::string client = argument(argc, argv, "--client-id");
    if (adapter.empty() || configuration.empty() || client.empty()) {
      throw std::runtime_error("--adapter, --config-json, and --client-id required");
    }
    const bool capture = flag(argc, argv, "--capture");
    const bool cta = flag(argc, argv, "--cta");
    Worker worker(adapter, configuration, client, !capture && !cta, cta);
    if (capture) {
      worker.warmup();
      worker.prepare_capture();
      using Warmup = int (*)(const char*);
      using SetFrame = int (*)(const char*, const char*, std::uint64_t);
      using Wait = int (*)(std::uint64_t);
      auto warmup = required_symbol<Warmup>(
          "pperf_agent_report_warmup_complete");
      auto set_frame = required_symbol<SetFrame>("pperf_agent_set_frame");
      auto wait = required_symbol<Wait>("pperf_agent_wait_capture_epoch");
      if (warmup(client.c_str()) != 0 ||
          set_frame(client.c_str(), "offline", 1) != 0) {
        throw std::runtime_error("agent warmup/frame marker failed");
      }
      write_status(argument(argc, argv, "--status"),
                   {{"state", "warmup_ready"}, {"identity", worker.identity()}});
      if (wait(300000) != 0) throw std::runtime_error("capture wait failed");
      worker.capture_launch();
      throw std::runtime_error("captured launch unexpectedly returned");
    }
    std::cout << json({{"status", "ready"}, {"identity", worker.identity()}})
                     .dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      const auto request = json::parse(line);
      if (request.at("operation") == "shutdown") break;
      json result;
      if (request.at("operation") == "run") {
        result = cta ? worker.cta_run(request.at("release_ns")) :
            worker.run(request.at("release_ns"),
                       request.value("enabled", true));
      } else {
        throw std::runtime_error("unknown worker operation");
      }
      std::cout << json({{"status", "ok"}, {"result", result}}).dump()
                << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
