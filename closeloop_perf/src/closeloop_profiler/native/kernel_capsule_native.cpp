#include "kernel_capsule_native.h"

#include "agent_state.h"
#include "nvbit_cta_abi.h"
#include "nvbit_cta_tracker_logic.h"

#include <cuda_runtime_api.h>
#include <cupti.h>
#include <cupti_activity.h>
#include <cupti_callbacks.h>
#include <cupti_checkpoint.h>
#include <cupti_driver_cbid.h>
#include <cupti_runtime_cbid.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

using json = nlohmann::json;
using NV::Cupti::Checkpoint::CUpti_Checkpoint;

namespace {

thread_local bool g_agent_cuda_call = false;
thread_local std::vector<std::string> g_owner_stack;
thread_local std::string g_pending_module_sha;

std::string pointer_string(const void* value) {
  std::ostringstream output;
  output << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(value);
  return output.str();
}

std::string hex_bytes(const void* bytes, std::size_t size) {
  const auto* value = static_cast<const unsigned char*>(bytes);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < size; ++index) {
    output << std::setw(2) << static_cast<unsigned int>(value[index]);
  }
  return output.str();
}

std::string sha256(const void* bytes, std::size_t size) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char*>(bytes), size, digest);
  return hex_bytes(digest, sizeof(digest));
}

std::string sha256(const std::string& value) {
  return sha256(value.data(), value.size());
}

std::uint64_t monotonic_ns() {
  timespec value{};
  clock_gettime(CLOCK_MONOTONIC, &value);
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

void sleep_until_monotonic(std::uint64_t release_ns) {
  timespec value{};
  value.tv_sec = static_cast<time_t>(release_ns / 1000000000ULL);
  value.tv_nsec = static_cast<long>(release_ns % 1000000000ULL);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &value, nullptr) ==
         EINTR) {}
}

struct AgentCudaScope {
  AgentCudaScope() : prior(g_agent_cuda_call) { g_agent_cuda_call = true; }
  ~AgentCudaScope() { g_agent_cuda_call = prior; }
  bool prior;
};

void require_cuda(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* text = nullptr;
  cuGetErrorString(result, &text);
  throw std::runtime_error(
      std::string("capsule_invalid: ") + operation + ": " +
      (text == nullptr ? std::to_string(result) : text));
}

void require_cupti(CUptiResult result, const char* operation) {
  if (result == CUPTI_SUCCESS) return;
  const char* text = nullptr;
  cuptiGetResultString(result, &text);
  throw std::runtime_error(
      std::string("capsule_invalid: ") + operation + ": " +
      (text == nullptr ? std::to_string(result) : text));
}

struct Allocation {
  CUdeviceptr base{};
  std::size_t size{};
  std::string id;
  std::string kind;
};

struct RegisteredOutput {
  CUdeviceptr pointer{};
  std::size_t size{};
  std::string name;
};

struct CapturedParameter {
  std::size_t index{};
  std::size_t offset{};
  std::vector<std::uint8_t> bytes;
  std::optional<std::string> allocation_id;
  std::optional<std::size_t> allocation_offset;
};

struct ActivityTiming {
  std::uint64_t start{};
  std::uint64_t end{};
};

struct NvbitCtaApi {
  pperf_nvbit_cta_prepare_fn prepare{};
  pperf_nvbit_cta_identify_next_launch_fn identify{};
  pperf_nvbit_cta_calibrate_fn calibrate{};
  pperf_nvbit_cta_collect_fn collect{};
  std::string version;
  std::string reason{"libpperf_nvbit_cta_tracker.so is not preloaded"};

  void probe() {
    const auto probe_function = reinterpret_cast<pperf_nvbit_cta_probe_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_probe"));
    prepare = reinterpret_cast<pperf_nvbit_cta_prepare_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_prepare"));
    identify = reinterpret_cast<pperf_nvbit_cta_identify_next_launch_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_identify_next_launch"));
    calibrate = reinterpret_cast<pperf_nvbit_cta_calibrate_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_calibrate"));
    collect = reinterpret_cast<pperf_nvbit_cta_collect_fn>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_collect"));
    const char* supplied_version = nullptr;
    std::vector<std::string> missing;
    for (const auto& value : {
             std::pair<const char*, const void*>{"probe", (const void*)probe_function},
             {"prepare", (const void*)prepare},
             {"identify", (const void*)identify},
             {"calibrate", (const void*)calibrate},
             {"collect", (const void*)collect},
         }) {
      if (value.second == nullptr) missing.emplace_back(value.first);
    }
    const int probe_result = probe_function == nullptr ? -1 :
        probe_function(PPERF_NVBIT_CTA_ABI_VERSION, &supplied_version);
    if (!missing.empty() || probe_result != 0 || supplied_version == nullptr) {
      prepare = nullptr;
      identify = nullptr;
      calibrate = nullptr;
      collect = nullptr;
      const char* unavailable = std::getenv(
          "PPERF_NVBIT_CTA_UNAVAILABLE_REASON");
      if (probe_function == nullptr && missing.size() == 5 &&
          unavailable != nullptr && *unavailable != '\0') {
        reason = unavailable;
        return;
      }
      std::ostringstream detail;
      detail << "NVBit CTA tracker ABI probe failed (result="
             << probe_result << ", missing=";
      for (std::size_t index = 0; index < missing.size(); ++index) {
        if (index != 0) detail << ',';
        detail << missing[index];
      }
      detail << ')';
      reason = detail.str();
      return;
    }
    version = supplied_version;
    reason.clear();
  }

  bool available() const {
    return prepare != nullptr;
  }
};

struct CapturedLaunch {
  std::string launch_id;
  std::string client_id;
  CUcontext context{};
  CUfunction function{};
  const void* runtime_stub{};
  CUstream stream{};
  int stream_priority{};
  std::string code_sha;
  std::string symbol;
  std::string api;
  std::string source_runtime_api;
  unsigned int grid[3]{1, 1, 1};
  unsigned int block[3]{1, 1, 1};
  std::optional<std::array<unsigned int, 3>> cluster;
  unsigned int shared_memory{};
  std::vector<CUlaunchAttribute> driver_attributes;
  std::vector<CapturedParameter> parameters;
  std::vector<std::vector<std::uint8_t>> stable_parameters;
  std::vector<void*> parameter_pointers;
  std::vector<std::uint8_t> packed_parameters;
  bool uses_packed_parameters{false};
  int submission_sequence_index{-1};
  std::int64_t client_release_offset_ns{};
  std::uint64_t source_host_launch_start_ns{};
  std::uint64_t source_host_launch_end_ns{};
  std::uint64_t source_gpu_start_ns{};
  std::uint64_t source_gpu_end_ns{};
  std::uint64_t driver_issue_start_ns{};
  std::uint64_t driver_issue_end_ns{};
  std::uint32_t capture_correlation{};
  std::uint32_t runtime_correlation{};
  std::uint32_t replay_correlation{};
  std::string owner;
  int occurrence_ordinal{-1};
  int sequence_index{-1};
  std::map<std::string, int> function_attributes;
  std::string captured_driver_fingerprint;

  json attribute_manifest() const {
    json values = json::array();
    for (const auto& attribute : driver_attributes) {
      values.push_back({
          {"id", static_cast<int>(attribute.id)},
          {"value_hex", hex_bytes(&attribute.value, sizeof(attribute.value))},
      });
    }
    return values;
  }

  json command_payload() const {
    json parameters_json = json::array();
    for (const auto& parameter : parameters) {
      parameters_json.push_back({
          {"index", parameter.index},
          {"offset", parameter.offset},
          {"size", parameter.bytes.size()},
          {"raw_bytes_hex",
           hex_bytes(parameter.bytes.data(), parameter.bytes.size())},
          {"pointer_allocation_id", parameter.allocation_id ?
               json(*parameter.allocation_id) : json(nullptr)},
          {"pointer_offset", parameter.allocation_offset ?
               json(*parameter.allocation_offset) : json(nullptr)},
      });
    }
    return {
        {"context_handle", pointer_string(context)},
        {"function_handle", pointer_string(function)},
        {"code_object_sha256", code_sha},
        {"captured_driver_api", api},
        {"replay_api", api},
        {"grid", {grid[0], grid[1], grid[2]}},
        {"block", {block[0], block[1], block[2]}},
        {"cluster", cluster ? json::array(
            {(*cluster)[0], (*cluster)[1], (*cluster)[2]}) : json(nullptr)},
        {"dynamic_shared_memory", shared_memory},
        {"stream_handle", pointer_string(stream)},
        {"stream_priority", stream_priority},
        {"function_attributes", function_attributes},
        {"launch_attributes", attribute_manifest()},
        {"parameter_mode", uses_packed_parameters ?
             "packed_extra" : "kernel_params"},
        {"parameters", parameters_json},
        {"packed_parameter_bytes_hex", uses_packed_parameters ?
             json(hex_bytes(packed_parameters.data(),
                            packed_parameters.size())) : json(nullptr)},
    };
  }

  json manifest() const {
    json values = json::array();
    for (const auto& parameter : parameters) {
      json value = {
          {"index", parameter.index},
          {"offset", parameter.offset},
          {"size", parameter.bytes.size()},
          {"raw_bytes_hex",
           hex_bytes(parameter.bytes.data(), parameter.bytes.size())},
          {"pointer_allocation_id",
           parameter.allocation_id ? json(*parameter.allocation_id) :
                                     json(nullptr)},
          {"pointer_offset",
           parameter.allocation_offset ? json(*parameter.allocation_offset) :
                                         json(nullptr)},
      };
      values.push_back(std::move(value));
    }
    json value = {
        {"launch_id", launch_id},
        {"client_id", client_id},
        {"context_handle", pointer_string(context)},
        {"function_handle", pointer_string(function)},
        {"runtime_stub_handle", runtime_stub == nullptr ? json(nullptr) :
                                  json(pointer_string(runtime_stub))},
        {"code_object_sha256", code_sha},
        {"symbol", symbol},
        {"symbol_sha256", sha256(symbol)},
        {"launch_api", api},
        {"source_runtime_api", source_runtime_api.empty() ? json(nullptr) :
                                      json(source_runtime_api)},
        {"captured_driver_api", api},
        {"replay_api", api},
        {"equivalence_level", "gpu_driver_command"},
        {"runtime_dispatch_preserved", false},
        {"driver_command_fingerprint", sha256(command_payload().dump())},
        {"grid", {grid[0], grid[1], grid[2]}},
        {"block", {block[0], block[1], block[2]}},
        {"cluster", nullptr},
        {"dynamic_shared_memory", shared_memory},
        {"stream_handle", pointer_string(stream)},
        {"stream_priority", stream_priority},
        {"function_attributes", function_attributes},
        {"launch_attributes", attribute_manifest()},
        {"parameters", values},
        {"parameter_mode", uses_packed_parameters ?
             "packed_extra" : "kernel_params"},
        {"packed_parameter_bytes_hex", uses_packed_parameters ?
             json(hex_bytes(packed_parameters.data(),
                            packed_parameters.size())) : json(nullptr)},
        {"source_runtime_correlation_id", source_runtime_api.empty() ?
             json(nullptr) : json(runtime_correlation)},
        {"nested_driver_correlation_id", capture_correlation},
        {"submission_policy", "common_epoch_dependency_burst"},
        {"submission_sequence_index", submission_sequence_index},
        {"client_release_offset_ns", client_release_offset_ns},
        {"source_host_launch_interval", {
             {"start_ns", source_host_launch_start_ns},
             {"end_ns", source_host_launch_end_ns}}},
        {"source_gpu_interval", {
             {"start_ns", source_gpu_start_ns},
             {"end_ns", source_gpu_end_ns}}},
        {"cupti_correlation_id", capture_correlation},
        {"framework_owner", owner.empty() ? json(nullptr) : json(owner)},
        {"frame_local_launch_occurrence_ordinal", occurrence_ordinal},
        {"frame_local_launch_sequence_index", sequence_index},
    };
    if (cluster) {
      value["cluster"] = {(*cluster)[0], (*cluster)[1], (*cluster)[2]};
    }
    return value;
  }

  std::string fingerprint() const {
    return sha256(command_payload().dump());
  }
};

struct RuntimeLaunch {
  bool active{false};
  bool driver_seen{false};
  std::string api;
  std::uint32_t correlation{};
  const void* stub{};
  void** parameters{};
  void** extra{};
  unsigned int grid[3]{1, 1, 1};
  unsigned int block[3]{1, 1, 1};
  unsigned int shared_memory{};
  CUstream stream{};
};

thread_local RuntimeLaunch g_runtime_launch;

struct CapturedEventOperation {
  CUevent event{};
  CUstream record_stream{};
  CUstream wait_stream{};
  unsigned int record_flags{};
  unsigned int wait_flags{};
  std::string record_api;
  std::string source_launch_id;
  std::string target_launch_id;

  json manifest() const {
    return {
        {"event_handle", pointer_string(event)},
        {"record_stream_handle", pointer_string(record_stream)},
        {"wait_stream_handle", pointer_string(wait_stream)},
        {"record_api", record_api},
        {"record_flags", record_flags},
        {"wait_flags", wait_flags},
        {"source_launch_id", source_launch_id},
        {"target_launch_id", target_launch_id},
    };
  }
};

struct DriverLaunchView {
  CUfunction function{};
  void** parameters{};
  void** extra{};
  unsigned int grid[3]{1, 1, 1};
  unsigned int block[3]{1, 1, 1};
  unsigned int shared_memory{};
  CUstream stream{};
  std::vector<CUlaunchAttribute> attributes;
  std::string api;
};

class KernelCapsuleAgent {
 public:
  static KernelCapsuleAgent& instance() {
    static KernelCapsuleAgent value;
    return value;
  }

  int initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return 1;
    try {
      client_id_ = required_environment("PPERF_KERNEL_CAPSULE_CLIENT_ID");
      socket_path_ = required_environment(
          "PPERF_KERNEL_CAPSULE_AGENT_SOCKET");
      specification_path_ = required_environment("PPERF_KERNEL_CAPSULE_SPEC");
      std::ifstream input(specification_path_);
      if (!input) throw std::runtime_error("capture specification unreadable");
      input >> specification_;
      if (specification_.value("schema", "") !=
          "kernel_capsule_capture_spec_v3") {
        throw std::runtime_error("capture specification schema mismatch");
      }
      build_expected_launches();
      nvbit_cta_.probe();
      state_.transition(pperf::AgentState::warming);
      subscribe_cupti();
      initialized_ = true;
      server_thread_ = std::thread([this] { server_loop(); });
      server_thread_.detach();
      lease_thread_ = std::thread([this] { lease_loop(); });
      lease_thread_.detach();
      return 1;
    } catch (const std::exception& error) {
      error_ = std::string("capsule_invalid: ") + error.what();
      return 0;
    }
  }

  int report_warmup(const char* model_id) {
    return guarded([&] {
      require_cupti(
          cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
          "warmup activity flush");
      std::lock_guard<std::mutex> lock(mutex_);
      if (model_id == nullptr || client_id_ != model_id) {
        throw std::runtime_error("warmup model identity mismatch");
      }
      state_.require(pperf::AgentState::warming, "warmup completion");
      state_.transition(pperf::AgentState::ready);
      condition_.notify_all();
    });
  }

  int set_frame(const char* model_id, const char* input_id,
                std::uint64_t stamp) {
    return guarded([&] {
      std::lock_guard<std::mutex> lock(mutex_);
      if (model_id == nullptr || input_id == nullptr ||
          client_id_ != model_id) {
        throw std::runtime_error("frame model identity mismatch");
      }
      frame_model_ = model_id;
      frame_input_ = input_id;
      frame_stamp_ = stamp;
      const auto expected_input = specification_.at("inference_inputs")
                                      .at(client_id_).get<std::string>();
      const auto expected_stamp = specification_.at("source_frame")
                                      .at("ros_header_timestamp_ns")
                                      .get<std::uint64_t>();
      if (frame_input_ != expected_input || frame_stamp_ != expected_stamp) {
        throw std::runtime_error("source frame/input identity mismatch");
      }
    });
  }

  int wait_capture_epoch(std::uint64_t timeout_ms) {
    return guarded([&] {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!condition_.wait_for(
              lock, std::chrono::milliseconds(timeout_ms), [&] {
                return state_.state() == pperf::AgentState::armed ||
                       state_.state() == pperf::AgentState::shutdown ||
                       !error_.empty();
              })) {
        throw std::runtime_error("capture epoch timeout");
      }
      if (!error_.empty()) throw std::runtime_error(error_);
      state_.require(pperf::AgentState::armed, "capture epoch");
      const auto release = capture_release_ns_;
      lock.unlock();
      sleep_until_monotonic(release);
      lock.lock();
      state_.transition(pperf::AgentState::capturing);
      capture_epoch_cupti_ns_ = 0;
      require_cupti(cuptiGetTimestamp(&capture_epoch_cupti_ns_),
                    "capture epoch timestamp");
      condition_.notify_all();
    });
  }

  int push_owner(const char* owner) {
    if (owner == nullptr) return 1;
    g_owner_stack.emplace_back(owner);
    return 0;
  }

  int pop_owner() {
    if (g_owner_stack.empty()) return 1;
    g_owner_stack.pop_back();
    return 0;
  }

  int frame_end(int success) {
    return guarded([&] {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_.state() != pperf::AgentState::frozen &&
          state_.state() != pperf::AgentState::shutdown) {
        std::ostringstream value;
        value << "capsule_invalid: frame ended before terminal capture"
              << " (success=" << success << ")";
        fail_locked(value.str());
        throw std::runtime_error(value.str());
      }
    });
  }

  int register_output(CUdeviceptr pointer, std::size_t size,
                      const char* name) {
    return guarded([&] {
      if (pointer == 0 || size == 0 || name == nullptr || *name == '\0') {
        throw std::runtime_error("invalid registered output");
      }
      std::lock_guard<std::mutex> lock(mutex_);
      state_.require(pperf::AgentState::warming, "output registration");
      const auto allocation = classify_pointer(pointer);
      if (!allocation || pointer + size < pointer ||
          pointer + size > allocation->base + allocation->size) {
        throw std::runtime_error("registered output is outside allocation");
      }
      registered_outputs_.push_back({pointer, size, name});
    });
  }

  void callback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                const void* callback_data) noexcept {
    try {
      if (g_agent_cuda_call) {
        if ((domain == CUPTI_CB_DOMAIN_DRIVER_API ||
             domain == CUPTI_CB_DOMAIN_RUNTIME_API) &&
            replay_launch_ != nullptr) {
          const auto* data =
              static_cast<const CUpti_CallbackData*>(callback_data);
          if (data->callbackSite == CUPTI_API_ENTER &&
              is_launch(domain, cbid)) {
            replay_launch_->replay_correlation = data->correlationId;
          }
        }
        return;
      }
      if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
        resource_callback(cbid, callback_data);
        return;
      }
      if (domain != CUPTI_CB_DOMAIN_DRIVER_API &&
          domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
      const auto* data = static_cast<const CUpti_CallbackData*>(callback_data);
      reject_unsupported(domain, cbid, data);
      dependency_callback(domain, data);
      allocation_callback(domain, cbid, data);
      module_callback(domain, cbid, data);
      if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        runtime_callback(cbid, data);
      } else {
        driver_callback(cbid, data);
      }
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(mutex_);
      fail_locked(error.what());
    }
  }

  void activity_record(const CUpti_Activity* record) {
    if (record->kind == CUPTI_ACTIVITY_KIND_FUNCTION) {
      const auto* function =
          reinterpret_cast<const CUpti_ActivityFunction*>(record);
      std::lock_guard<std::mutex> lock(activity_mutex_);
      const auto module = module_id_hashes_.find(function->moduleId);
      if (module != module_id_hashes_.end() && function->name != nullptr) {
        symbol_code_hashes_[function->name].insert(module->second);
      }
      return;
    }
    if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL &&
        record->kind != CUPTI_ACTIVITY_KIND_KERNEL) return;
    const auto* kernel = reinterpret_cast<const CUpti_ActivityKernel9*>(record);
    std::lock_guard<std::mutex> lock(activity_mutex_);
    activity_[kernel->correlationId] = {kernel->start, kernel->end};
  }

 private:
  KernelCapsuleAgent() = default;

  static std::string required_environment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
      throw std::runtime_error(std::string(name) + " is not set");
    }
    return value;
  }

  template <typename Function>
  int guarded(Function function) noexcept {
    try {
      function();
      return 0;
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(mutex_);
      fail_locked(std::string("capsule_invalid: ") + error.what());
      std::fprintf(stderr, "pperf capsule agent: %s\n", error.what());
      return 1;
    }
  }

  void fail_locked(const std::string& error) {
    if (error_.empty()) error_ = error;
    condition_.notify_all();
  }

  void build_expected_launches() {
    std::vector<json> all;
    all.push_back(specification_.at("aggressor"));
    for (const auto& value : specification_.at("victim_kernels")) {
      all.push_back(value);
    }
    int victim_index = 0;
    int submission_index = 0;
    for (std::size_t index = 0; index < all.size(); ++index) {
      json value = all[index];
      const auto model = value.value("model_identity", "");
      if (model != client_id_) {
        if (index != 0) ++victim_index;
        continue;
      }
      value["launch_id"] = index == 0 ? "aggressor" :
          "victim-" + std::to_string(victim_index++);
      value["submission_sequence_index"] = submission_index++;
      value["client_release_offset_ns"] =
          specification_.at("client_release_offsets_ns").at(client_id_);
      value["source_host_launch_interval"] = {
          {"start_ns", value.at("launch_start_ns")},
          {"end_ns", value.at("launch_end_ns")},
      };
      value["source_gpu_interval"] = value.contains("source_gpu_interval") ?
          value.at("source_gpu_interval") : json{
              {"start_ns", value.at("start_ns")},
              {"end_ns", value.at("end_ns")},
          };
      expected_.push_back(std::move(value));
    }
    if (expected_.empty()) {
      throw std::runtime_error("capture specification has no client launch");
    }
  }

  void subscribe_cupti() {
    require_cupti(cuptiSubscribe(&subscriber_, callback_entry, this),
                  "cuptiSubscribe");
    for (auto domain : {
             CUPTI_CB_DOMAIN_DRIVER_API,
             CUPTI_CB_DOMAIN_RUNTIME_API,
             CUPTI_CB_DOMAIN_RESOURCE,
             CUPTI_CB_DOMAIN_SYNCHRONIZE,
         }) {
      require_cupti(cuptiEnableDomain(1, subscriber_, domain),
                    "cuptiEnableDomain");
    }
    require_cupti(cuptiActivityRegisterCallbacks(
                      activity_buffer_requested, activity_buffer_completed),
                  "cuptiActivityRegisterCallbacks");
    require_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
                  "cuptiActivityEnable kernels");
    require_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_FUNCTION),
                  "cuptiActivityEnable functions");
  }

  static void CUPTIAPI callback_entry(void* user_data,
                                      CUpti_CallbackDomain domain,
                                      CUpti_CallbackId cbid,
                                      const void* callback_data) {
    static_cast<KernelCapsuleAgent*>(user_data)->callback(
        domain, cbid, callback_data);
  }

  static void CUPTIAPI activity_buffer_requested(
      std::uint8_t** buffer, std::size_t* size, std::size_t* max_records) {
    constexpr std::size_t kSize = 1024 * 1024;
    void* value = nullptr;
    if (posix_memalign(&value, 8, kSize) != 0) value = nullptr;
    *buffer = static_cast<std::uint8_t*>(value);
    *size = value == nullptr ? 0 : kSize;
    *max_records = 0;
  }

  static void CUPTIAPI activity_buffer_completed(
      CUcontext, std::uint32_t, std::uint8_t* buffer, std::size_t,
      std::size_t valid_size) {
    if (buffer == nullptr) return;
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) ==
           CUPTI_SUCCESS) {
      instance().activity_record(record);
    }
    std::free(buffer);
  }

  static bool is_launch(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
      return cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
             cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
             cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
             cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz ||
             cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
             cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz;
    }
    return cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
           cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000 ||
           cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060 ||
           cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060 ||
           cbid ==
               CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 ||
           cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000;
  }

  void reject_unsupported(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                          const CUpti_CallbackData* data) {
    if (data->callbackSite != CUPTI_API_ENTER) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto state = state_.state();
    if (state != pperf::AgentState::armed &&
        state != pperf::AgentState::capturing) return;
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
        (cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
         cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz ||
         cbid == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture ||
         cbid == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz ||
         cbid == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2 ||
         cbid == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz ||
         cbid == CUPTI_DRIVER_TRACE_CBID_cuImportExternalMemory)) {
      throw std::runtime_error(
          "capsule_invalid: graph, stream-capture, or external-memory API");
    }
    if (checkpoint_generation_ <= 0) return;
    const std::string name = data->functionName == nullptr ? "" :
        data->functionName;
    const bool event_operation =
        name.find("EventRecord") != std::string::npos ||
        name.find("StreamWaitEvent") != std::string::npos;
    const bool passive_query =
        name.find("Get") != std::string::npos ||
        name.find("Query") != std::string::npos ||
        name.find("Occupancy") != std::string::npos ||
        name == "cudaPeekAtLastError";
    if (name.find("Memcpy") != std::string::npos ||
        name.find("Memset") != std::string::npos) {
      throw std::runtime_error(
          "capsule_invalid: post-checkpoint memcpy/memset dependency");
    }
    if (!is_launch(domain, cbid) && !event_operation &&
        !passive_query) {
      throw std::runtime_error(
          "capsule_invalid: foreign CUDA work after checkpoint: " + name);
    }
  }

  void dependency_callback(CUpti_CallbackDomain domain,
                           const CUpti_CallbackData* data) {
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API ||
        data->callbackSite != CUPTI_API_ENTER ||
        checkpoint_generation_ <= 0) return;
    const std::string name = data->functionName == nullptr ? "" :
        data->functionName;
    if (name.find("EventRecord") != std::string::npos) {
      CUevent event = nullptr;
      CUstream stream = nullptr;
      unsigned int flags = 0;
      std::string api = "cuEventRecord";
      if (name.find("EventRecordWithFlags") != std::string::npos) {
        const auto* value =
            static_cast<const cuEventRecordWithFlags_params*>(
                data->functionParams);
        event = value->hEvent;
        stream = value->hStream;
        flags = value->flags;
        api = "cuEventRecordWithFlags";
      } else {
        const auto* value = static_cast<const cuEventRecord_params*>(
            data->functionParams);
        event = value->hEvent;
        stream = value->hStream;
      }
      const auto source = std::find_if(
          captured_.rbegin(), captured_.rend(), [&](const CapturedLaunch& item) {
            return item.stream == stream;
          });
      if (source == captured_.rend()) {
        throw std::runtime_error(
            "capsule_invalid: event record has external producer");
      }
      CapturedEventOperation operation;
      operation.event = event;
      operation.record_stream = stream;
      operation.record_flags = flags;
      operation.record_api = api;
      operation.source_launch_id = source->launch_id;
      const auto index = event_operations_.size();
      event_operations_.push_back(std::move(operation));
      recorded_events_[event] = index;
    } else if (name.find("StreamWaitEvent") != std::string::npos) {
      const auto* value = static_cast<const cuStreamWaitEvent_params*>(
          data->functionParams);
      const auto found = recorded_events_.find(value->hEvent);
      if (found == recorded_events_.end()) {
        throw std::runtime_error(
            "capsule_invalid: event wait has external producer");
      }
      auto& operation = event_operations_.at(found->second);
      operation.wait_stream = value->hStream;
      operation.wait_flags = value->Flags;
      pending_event_waits_[value->hStream].push_back(found->second);
    }
  }

  static bool successful(const CUpti_CallbackData* data) {
    return data->functionReturnValue != nullptr &&
           *static_cast<const CUresult*>(data->functionReturnValue) ==
               CUDA_SUCCESS;
  }

  void allocation_callback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                           const CUpti_CallbackData* data) {
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API ||
        data->callbackSite != CUPTI_API_EXIT || !successful(data)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto add = [&](CUdeviceptr base, std::size_t size, const char* kind) {
      Allocation value{base, size,
                       std::string("allocation-") +
                           std::to_string(++allocation_generation_), kind};
      allocations_[base] = std::move(value);
    };
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2) {
      const auto* value = static_cast<const cuMemAlloc_v2_params*>(
          data->functionParams);
      add(*value->dptr, value->bytesize, "device");
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz) {
      const auto* value = static_cast<const cuMemAllocAsync_params*>(
          data->functionParams);
      add(*value->dptr, value->bytesize, "async");
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz) {
      const auto* value =
          static_cast<const cuMemAllocFromPoolAsync_params*>(
              data->functionParams);
      add(*value->dptr, value->bytesize, "pool");
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemMap) {
      const auto* value = static_cast<const cuMemMap_params*>(
          data->functionParams);
      add(value->ptr, value->size, "vmm");
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2) {
      const auto* value = static_cast<const cuMemFree_v2_params*>(
          data->functionParams);
      allocations_.erase(value->dptr);
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz) {
      const auto* value = static_cast<const cuMemFreeAsync_params*>(
          data->functionParams);
      allocations_.erase(value->dptr);
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemUnmap) {
      const auto* value = static_cast<const cuMemUnmap_params*>(
          data->functionParams);
      allocations_.erase(value->ptr);
    }
  }

  void resource_callback(CUpti_CallbackId cbid, const void* callback_data) {
    if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      const auto* resource =
          static_cast<const CUpti_ResourceData*>(callback_data);
      const auto* value = static_cast<const CUpti_ModuleResourceData*>(
          resource->resourceDescriptor);
      if (value != nullptr) {
        const std::string digest = sha256(value->pCubin, value->cubinSize);
        g_pending_module_sha = digest;
        std::lock_guard<std::mutex> lock(activity_mutex_);
        module_id_hashes_[value->moduleId] = digest;
      }
    } else if (cbid == CUPTI_CBID_RESOURCE_STREAM_CREATED) {
      const auto* value = static_cast<const CUpti_ResourceData*>(callback_data);
      std::lock_guard<std::mutex> lock(mutex_);
      stream_priorities_[value->resourceHandle.stream] = 0;
    }
  }

  void module_callback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                       const CUpti_CallbackData* data) {
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API ||
        data->callbackSite != CUPTI_API_EXIT || !successful(data)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData) {
      const auto* value = static_cast<const cuModuleLoadData_params*>(
          data->functionParams);
      if (value->module != nullptr && !g_pending_module_sha.empty()) {
        module_hashes_[*value->module] = g_pending_module_sha;
        g_pending_module_sha.clear();
      }
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx) {
      const auto* value = static_cast<const cuModuleLoadDataEx_params*>(
          data->functionParams);
      if (value->module != nullptr && !g_pending_module_sha.empty()) {
        module_hashes_[*value->module] = g_pending_module_sha;
        g_pending_module_sha.clear();
      }
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction) {
      const auto* value = static_cast<const cuModuleGetFunction_params*>(
          data->functionParams);
      if (value->hfunc != nullptr) function_modules_[*value->hfunc] = value->hmod;
    }
  }

  void runtime_callback(CUpti_CallbackId cbid,
                        const CUpti_CallbackData* data) {
    if (!is_launch(CUPTI_CB_DOMAIN_RUNTIME_API, cbid)) return;
    if (data->callbackSite == CUPTI_API_ENTER) {
      if (g_runtime_launch.active) {
        throw std::runtime_error("capsule_invalid: nested runtime launch");
      }
      RuntimeLaunch value;
      value.active = true;
      value.correlation = data->correlationId;
      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
          cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000) {
        const auto* params =
            static_cast<const cudaLaunchKernel_v7000_params*>(
                data->functionParams);
        value.api = "cudaLaunchKernel_v7000";
        value.stub = params->func;
        value.parameters = params->args;
        value.grid[0] = params->gridDim.x;
        value.grid[1] = params->gridDim.y;
        value.grid[2] = params->gridDim.z;
        value.block[0] = params->blockDim.x;
        value.block[1] = params->blockDim.y;
        value.block[2] = params->blockDim.z;
        value.shared_memory = params->sharedMem;
        value.stream = reinterpret_cast<CUstream>(params->stream);
      } else if (
          cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060 ||
          cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060) {
        const auto* params =
            static_cast<const cudaLaunchKernelExC_v11060_params*>(
                data->functionParams);
        value.api = "cudaLaunchKernelExC_v11060";
        value.stub = params->func;
        value.parameters = params->args;
        value.grid[0] = params->config->gridDim.x;
        value.grid[1] = params->config->gridDim.y;
        value.grid[2] = params->config->gridDim.z;
        value.block[0] = params->config->blockDim.x;
        value.block[1] = params->config->blockDim.y;
        value.block[2] = params->config->blockDim.z;
        value.shared_memory = params->config->dynamicSmemBytes;
        value.stream = reinterpret_cast<CUstream>(params->config->stream);
      } else {
        const auto* params =
            static_cast<const cudaLaunchCooperativeKernel_v9000_params*>(
                data->functionParams);
        value.api = "cudaLaunchCooperativeKernel";
        value.stub = params->func;
        value.parameters = params->args;
        value.grid[0] = params->gridDim.x;
        value.grid[1] = params->gridDim.y;
        value.grid[2] = params->gridDim.z;
        value.block[0] = params->blockDim.x;
        value.block[1] = params->blockDim.y;
        value.block[2] = params->blockDim.z;
        value.shared_memory = params->sharedMem;
        value.stream = reinterpret_cast<CUstream>(params->stream);
      }
      g_runtime_launch = std::move(value);
    } else {
      if (g_runtime_launch.active && !g_runtime_launch.driver_seen) {
        throw std::runtime_error(
            "capsule_invalid: runtime launch has no verifiable CUfunction");
      }
      if (g_runtime_launch.active && g_runtime_launch.driver_seen) {
        finish_launch(active_selected_correlation_);
      }
      g_runtime_launch = RuntimeLaunch{};
    }
  }

  static DriverLaunchView driver_view(CUpti_CallbackId cbid,
                                      const void* parameters) {
    DriverLaunchView view;
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
      const auto* value = static_cast<const cuLaunchKernel_params*>(parameters);
      view.function = value->f;
      view.parameters = value->kernelParams;
      view.extra = value->extra;
      view.grid[0] = value->gridDimX;
      view.grid[1] = value->gridDimY;
      view.grid[2] = value->gridDimZ;
      view.block[0] = value->blockDimX;
      view.block[1] = value->blockDimY;
      view.block[2] = value->blockDimZ;
      view.shared_memory = value->sharedMemBytes;
      view.stream = value->hStream;
      view.api = "cuLaunchKernel";
    } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz) {
      const auto* value =
          static_cast<const cuLaunchKernelEx_params*>(parameters);
      view.function = value->f;
      view.parameters = value->kernelParams;
      view.extra = value->extra;
      view.grid[0] = value->config->gridDimX;
      view.grid[1] = value->config->gridDimY;
      view.grid[2] = value->config->gridDimZ;
      view.block[0] = value->config->blockDimX;
      view.block[1] = value->config->blockDimY;
      view.block[2] = value->config->blockDimZ;
      view.shared_memory = value->config->sharedMemBytes;
      view.stream = value->config->hStream;
      view.attributes.assign(value->config->attrs,
                             value->config->attrs + value->config->numAttrs);
      view.api = "cuLaunchKernelEx";
    } else {
      const auto* value =
          static_cast<const cuLaunchCooperativeKernel_params*>(parameters);
      view.function = value->f;
      view.parameters = value->kernelParams;
      view.grid[0] = value->gridDimX;
      view.grid[1] = value->gridDimY;
      view.grid[2] = value->gridDimZ;
      view.block[0] = value->blockDimX;
      view.block[1] = value->blockDimY;
      view.block[2] = value->blockDimZ;
      view.shared_memory = value->sharedMemBytes;
      view.stream = value->hStream;
      view.api = "cuLaunchCooperativeKernel";
    }
    return view;
  }

  void driver_callback(CUpti_CallbackId cbid,
                       const CUpti_CallbackData* data) {
    if (!is_launch(CUPTI_CB_DOMAIN_DRIVER_API, cbid)) return;
    if (data->callbackSite == CUPTI_API_ENTER) {
      DriverLaunchView view = driver_view(cbid, data->functionParams);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        known_streams_[data->context].insert(view.stream);
      }
      if (g_runtime_launch.active) {
        g_runtime_launch.driver_seen = true;
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_.state() == pperf::AgentState::warming) {
          const auto previous = runtime_target_functions_.find(
              g_runtime_launch.stub);
          if (previous != runtime_target_functions_.end() &&
              previous->second != view.function) {
            throw std::runtime_error(
                "capsule_invalid: runtime target maps to multiple functions");
          }
          runtime_target_functions_[g_runtime_launch.stub] = view.function;
        }
      }
      capture_launch(view, data);
    } else if (!g_runtime_launch.active) {
      finish_launch(data->correlationId);
    }
  }

  bool matcher_equal(const DriverLaunchView& view, const std::string& api,
                     const std::string& symbol, const json& expected) const {
    const auto matcher = expected.at("capture_matcher");
    const std::string exact_symbol = matcher.value(
        "mangled_symbol", matcher.at("symbol").get<std::string>());
    return matcher.at("launch_api") == api &&
           exact_symbol == symbol &&
           matcher.at("grid") == json::array({
               view.grid[0], view.grid[1], view.grid[2]}) &&
           matcher.at("block") == json::array({
               view.block[0], view.block[1], view.block[2]}) &&
           matcher.at("dynamic_shared_memory") == view.shared_memory &&
           (matcher.value("framework_owner", "") == "" ||
            matcher.value("framework_owner", "") == current_owner());
  }

  std::string current_owner() const {
    return g_owner_stack.empty() ? std::string() : g_owner_stack.back();
  }

  std::optional<Allocation> classify_pointer(std::uint64_t value) const {
    auto upper = allocations_.upper_bound(static_cast<CUdeviceptr>(value));
    if (upper == allocations_.begin()) return std::nullopt;
    --upper;
    const auto& allocation = upper->second;
    if (value < allocation.base ||
        value - allocation.base >= allocation.size) return std::nullopt;
    return allocation;
  }

  void capture_parameters(CapturedLaunch& launch, void** kernel_parameters,
                          void** extra) {
#if CUDA_VERSION < 12040
    (void)launch;
    (void)kernel_parameters;
    (void)extra;
    throw std::runtime_error("capsule_invalid: cuFuncGetParamInfo unavailable");
#else
    const std::uint8_t* packed = nullptr;
    std::size_t packed_size = 0;
    if (kernel_parameters == nullptr && extra != nullptr) {
      for (std::size_t index = 0; extra[index] != CU_LAUNCH_PARAM_END;
           index += 2) {
        if (extra[index] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
          packed = static_cast<const std::uint8_t*>(extra[index + 1]);
        } else if (extra[index] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
          packed_size = *static_cast<const std::size_t*>(extra[index + 1]);
        } else {
          throw std::runtime_error(
              "capsule_invalid: unsupported packed launch token");
        }
      }
      if (packed == nullptr || packed_size == 0) {
        throw std::runtime_error(
            "capsule_invalid: incomplete packed launch buffer");
      }
      launch.packed_parameters.assign(packed, packed + packed_size);
      launch.uses_packed_parameters = true;
    }
    for (std::size_t index = 0;; ++index) {
      std::size_t offset = 0;
      std::size_t size = 0;
      CUresult result = cuFuncGetParamInfo(
          launch.function, index, &offset, &size);
      if (result == CUDA_ERROR_INVALID_VALUE) break;
      require_cuda(result, "cuFuncGetParamInfo");
      const void* source = nullptr;
      if (kernel_parameters != nullptr) {
        source = kernel_parameters[index];
      } else if (offset + size <= launch.packed_parameters.size()) {
        source = launch.packed_parameters.data() + offset;
      }
      if (source == nullptr) {
        throw std::runtime_error("capsule_invalid: null kernel parameter");
      }
      CapturedParameter parameter;
      parameter.index = index;
      parameter.offset = offset;
      parameter.bytes.resize(size);
      std::memcpy(parameter.bytes.data(), source, size);
      if (size == sizeof(std::uint64_t)) {
        std::uint64_t possible_pointer = 0;
        std::memcpy(&possible_pointer, source, sizeof(possible_pointer));
        auto allocation = classify_pointer(possible_pointer);
        if (allocation) {
          parameter.allocation_id = allocation->id;
          parameter.allocation_offset =
              possible_pointer - allocation->base;
        }
      }
      launch.parameters.push_back(std::move(parameter));
    }
    launch.stable_parameters.reserve(launch.parameters.size());
    launch.parameter_pointers.reserve(launch.parameters.size());
    for (const auto& parameter : launch.parameters) {
      launch.stable_parameters.push_back(parameter.bytes);
    }
    for (auto& parameter : launch.stable_parameters) {
      launch.parameter_pointers.push_back(parameter.data());
    }
#endif
  }

  void capture_launch(const DriverLaunchView& driver,
                      const CUpti_CallbackData* data) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_.state() != pperf::AgentState::capturing) return;
    const std::string source_api = g_runtime_launch.active ?
        g_runtime_launch.api : driver.api;
    const std::string symbol = data->symbolName == nullptr ? "" :
        data->symbolName;
    ++frame_sequence_index_;
    std::map<std::string, int> launch_occurrences;
    std::set<std::string> counted_matchers;
    for (std::size_t index = 0; index < expected_.size(); ++index) {
      if (!matcher_equal(driver, source_api, symbol, expected_[index])) continue;
      const std::string matcher_key = expected_[index].at("capture_matcher").dump();
      if (counted_matchers.insert(matcher_key).second) {
        launch_occurrences.emplace(
            matcher_key, matcher_occurrences_[matcher_key]++);
      }
    }
    const std::size_t selected_index = captured_.size();
    const auto& expected = expected_[selected_index];
    const int expected_sequence = expected.at(
        "frame_local_launch_sequence_index").get<int>();
    if (frame_sequence_index_ < expected_sequence) return;
    if (frame_sequence_index_ > expected_sequence) {
      std::ostringstream message;
      message << "capsule_invalid: selected launch sequence was skipped"
              << " client=" << client_id_
              << " launch_id=" << expected.value("launch_id", "unknown")
              << " expected_sequence=" << expected_sequence
              << " observed_sequence=" << frame_sequence_index_;
      throw std::runtime_error(message.str());
    }
    if (!matcher_equal(driver, source_api, symbol, expected)) {
      const auto& matcher = expected.at("capture_matcher");
      std::ostringstream message;
      message << "capsule_invalid: selected launch fingerprint mismatch"
              << " client=" << client_id_
              << " launch_id=" << expected.value("launch_id", "unknown")
              << " sequence=" << expected_sequence
              << " expected_api=" << matcher.value("launch_api", "")
              << " observed_api=" << source_api
              << " expected_symbol=" << matcher.value(
                     "mangled_symbol", matcher.value("symbol", ""))
              << " observed_symbol=" << symbol
              << " expected_grid=" << matcher.at("grid").dump()
              << " observed_grid=" << json::array({
                     driver.grid[0], driver.grid[1], driver.grid[2]}).dump()
              << " expected_block=" << matcher.at("block").dump()
              << " observed_block=" << json::array({
                     driver.block[0], driver.block[1], driver.block[2]}).dump()
              << " expected_shared=" << matcher.at(
                     "dynamic_shared_memory")
              << " observed_shared=" << driver.shared_memory
              << " expected_owner=" << matcher.value(
                     "framework_owner", "")
              << " observed_owner=" << current_owner();
      throw std::runtime_error(message.str());
    }
    const std::string matcher_key = expected.at("capture_matcher").dump();
    const int occurrence = launch_occurrences.at(matcher_key);
    const int expected_occurrence = expected.at(
        "frame_local_launch_occurrence_ordinal").get<int>();
    if (occurrence != expected_occurrence) {
      std::ostringstream message;
      message << "capsule_invalid: selected launch occurrence mismatch"
              << " client=" << client_id_
              << " launch_id=" << expected.value("launch_id", "unknown")
              << " sequence=" << expected_sequence
              << " expected_occurrence=" << expected_occurrence
              << " observed_occurrence=" << occurrence;
      throw std::runtime_error(message.str());
    }
    CapturedLaunch launch;
    launch.launch_id = expected.at("launch_id");
    launch.client_id = client_id_;
    launch.context = data->context;
    launch.function = driver.function;
    launch.runtime_stub = g_runtime_launch.active ? g_runtime_launch.stub : nullptr;
    launch.stream = driver.stream;
    launch.api = driver.api;
    launch.source_runtime_api = g_runtime_launch.active ?
        g_runtime_launch.api : std::string();
    launch.symbol = symbol;
    std::copy(std::begin(driver.grid), std::end(driver.grid), launch.grid);
    std::copy(std::begin(driver.block), std::end(driver.block), launch.block);
    launch.shared_memory = driver.shared_memory;
    launch.driver_attributes = driver.attributes;
    for (const auto& attribute : driver.attributes) {
      if (attribute.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
        launch.cluster = std::array<unsigned int, 3>{
            attribute.value.clusterDim.x,
            attribute.value.clusterDim.y,
            attribute.value.clusterDim.z,
        };
      }
    }
    launch.submission_sequence_index = expected.at(
        "submission_sequence_index");
    launch.client_release_offset_ns = expected.at(
        "client_release_offset_ns");
    const auto& source_host = expected.at("source_host_launch_interval");
    launch.source_host_launch_start_ns = source_host.at("start_ns");
    launch.source_host_launch_end_ns = source_host.at("end_ns");
    const auto& source_gpu = expected.at("source_gpu_interval");
    launch.source_gpu_start_ns = source_gpu.at("start_ns");
    launch.source_gpu_end_ns = source_gpu.at("end_ns");
    launch.capture_correlation = data->correlationId;
    launch.runtime_correlation = g_runtime_launch.active ?
        g_runtime_launch.correlation : 0;
    launch.owner = current_owner();
    launch.occurrence_ordinal = expected.at(
        "frame_local_launch_occurrence_ordinal");
    launch.sequence_index = expected.at("frame_local_launch_sequence_index");
    AgentCudaScope bypass;
    require_cuda(cuStreamGetPriority(launch.stream, &launch.stream_priority),
                 "cuStreamGetPriority");
    if (launch.stream_priority != expected.at("stream_priority").get<int>()) {
      throw std::runtime_error(
          "capsule_invalid: selected stream priority changed");
    }
    const int source_stream_id = expected.at("stream_id").get<int>();
    const auto mapped_stream = source_stream_handles_.find(source_stream_id);
    if (mapped_stream != source_stream_handles_.end() &&
        mapped_stream->second != launch.stream) {
      throw std::runtime_error(
          "capsule_invalid: source stream maps to multiple live streams");
    }
    for (const auto& item : source_stream_handles_) {
      if (item.first != source_stream_id && item.second == launch.stream) {
        throw std::runtime_error(
            "capsule_invalid: distinct source streams collapsed at capture");
      }
    }
    source_stream_handles_[source_stream_id] = launch.stream;
    CUmodule module = nullptr;
    require_cuda(cuFuncGetModule(&module, launch.function), "cuFuncGetModule");
    for (const auto& item : {
             std::pair<int, CUfunction_attribute>{
                 CU_FUNC_ATTRIBUTE_NUM_REGS,
                 CU_FUNC_ATTRIBUTE_NUM_REGS},
             {CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
              CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES},
             {CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
              CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK},
             {CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
              CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES},
         }) {
      int value = 0;
      require_cuda(cuFuncGetAttribute(&value, item.second, launch.function),
                   "cuFuncGetAttribute");
      launch.function_attributes[std::to_string(item.first)] = value;
    }
    if (launch.runtime_stub != nullptr) {
      const auto warmup_function = runtime_target_functions_.find(
          launch.runtime_stub);
      if (warmup_function == runtime_target_functions_.end() ||
          warmup_function->second != launch.function) {
        throw std::runtime_error(
            "capsule_invalid: runtime target changed after warmup");
      }
    }
    auto module_hash = module_hashes_.find(module);
    if (module_hash != module_hashes_.end()) {
      launch.code_sha = module_hash->second;
    } else {
      std::lock_guard<std::mutex> activity_lock(activity_mutex_);
      const auto symbol_hashes = symbol_code_hashes_.find(symbol);
      if (symbol_hashes == symbol_code_hashes_.end() ||
          symbol_hashes->second.size() != 1) {
        const auto count = symbol_hashes == symbol_code_hashes_.end() ? 0 :
            symbol_hashes->second.size();
        throw std::runtime_error(
            "capsule_invalid: launch code object cannot be resolved exactly "
            "for symbol=" + symbol + " candidates=" +
            std::to_string(count));
      }
      launch.code_sha = *symbol_hashes->second.begin();
      module_hashes_[module] = launch.code_sha;
    }
    if (captured_.empty()) {
      checkpoint_context_ = launch.context;
      const auto known = known_streams_[launch.context];
      lock.unlock();
      {
        AgentCudaScope bypass;
        for (CUstream stream : known) {
          require_cuda(cuStreamSynchronize(stream),
                       "pre-capture stream synchronize");
        }
        checkpoint_.reset(pperf_checkpoint_create(launch.context));
        if (!checkpoint_ || pperf_checkpoint_save(checkpoint_.get()) !=
                                CUPTI_SUCCESS) {
          throw std::runtime_error("capsule_invalid: checkpoint save failed");
        }
      }
      lock.lock();
      checkpoint_generation_ = 1;
    } else if (launch.context != checkpoint_context_) {
      throw std::runtime_error(
          "capsule_invalid: multiple client contexts selected");
    }
    capture_parameters(launch, driver.parameters, driver.extra);
    launch.captured_driver_fingerprint = sha256(
        launch.command_payload().dump());
    const auto pending = pending_event_waits_.find(launch.stream);
    if (pending != pending_event_waits_.end()) {
      for (std::size_t index : pending->second) {
        event_operations_.at(index).target_launch_id = launch.launch_id;
      }
      pending_event_waits_.erase(pending);
    }
    active_selected_correlation_ = data->correlationId;
    captured_.push_back(std::move(launch));
  }

  void finish_launch(std::uint32_t correlation) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_.state() != pperf::AgentState::capturing ||
        correlation != active_selected_correlation_ ||
        captured_.size() != expected_.size()) return;
    if (!pending_event_waits_.empty() || std::any_of(
            event_operations_.begin(), event_operations_.end(),
            [](const CapturedEventOperation& operation) {
              return operation.target_launch_id.empty();
            })) {
      throw std::runtime_error(
          "capsule_invalid: unresolved or external event dependency");
    }
    state_.transition(pperf::AgentState::quiescing);
    std::set<CUstream> streams;
    for (const auto& launch : captured_) streams.insert(launch.stream);
    terminal_streams_ = streams;
    lock.unlock();
    {
      AgentCudaScope bypass;
      for (CUstream stream : streams) {
        require_cuda(cuStreamSynchronize(stream),
                     "terminal stream synchronize");
      }
      require_cupti(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
                    "capture activity flush");
    }
    lock.lock();
    state_.transition(pperf::AgentState::frozen);
    condition_.notify_all();
    condition_.wait(lock, [&] {
      return state_.state() == pperf::AgentState::shutdown || !error_.empty();
    });
  }

  void update_lease(const json& request) {
    const auto lease_id = request.value("lease_id", "");
    const auto expires = request.value(
        "lease_expires_monotonic_ns", std::uint64_t{0});
    if (lease_id.empty() || expires <= monotonic_ns()) {
      throw std::runtime_error("capsule_invalid: coordinator lease missing");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!lease_id_.empty() && lease_id != lease_id_) {
      throw std::runtime_error("capsule_invalid: coordinator lease changed");
    }
    lease_id_ = lease_id;
    lease_expiry_ns_ = expires;
  }

  json capabilities() const {
    return {{"collectors", {
        {"cupti_activity", {{"available", true}, {"version", CUDA_VERSION}}},
        {"cuda_launch_capture", {{"available", true}, {"version", "v3"}}},
        {"cupti_checkpoint", {{"available", true}, {"version", CUDA_VERSION}}},
        {"cupti_pm", {{"available", false},
            {"reason", "CUPTI PM startup compatibility probe not enabled"}}},
        {"nvbit_cta", {{"available", nvbit_cta_.available()},
            {"version", nvbit_cta_.available() ?
                json(nvbit_cta_.version) : json(nullptr)},
            {"reason", nvbit_cta_.available() ?
                json(nullptr) : json(nvbit_cta_.reason)}}},
        {"ncu_mps", {{"available", false},
            {"reason", "requires relaunch under ncu --mps client/control"}}},
    }}};
  }

  json capture_rpc(const json& request) {
    std::unique_lock<std::mutex> lock(mutex_);
    state_.require(pperf::AgentState::ready, "capture");
    const auto& supplied = request.at("capture_specification");
    if (sha256(supplied.dump()) != sha256(specification_.dump())) {
      throw std::runtime_error("capsule_invalid: capture specification changed");
    }
    capture_release_ns_ = request.at("coordinator_capture_release_ns");
    state_.transition(pperf::AgentState::armed);
    condition_.notify_all();
    condition_.wait(lock, [&] {
      return state_.state() == pperf::AgentState::frozen || !error_.empty();
    });
    if (!error_.empty()) throw std::runtime_error(error_);
    json launches = json::array();
    for (const auto& launch : captured_) launches.push_back(launch.manifest());
    json streams = json::array();
    for (CUstream stream : terminal_streams_) {
      streams.push_back(pointer_string(stream));
    }
    json client = {
        {"client_id", client_id_},
        {"server_pid", 0},
        {"client_pid", getpid()},
        {"context_handle", pointer_string(checkpoint_context_)},
        {"active_thread_percentage", std::atoi(std::getenv(
             "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") == nullptr ? "100" :
             std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))},
        {"client_priority", 0},
        {"connection_count", 1},
        {"checkpoint_id", client_id_ + "-checkpoint-1"},
        {"checkpoint_saved", true},
        {"checkpoint_generation", checkpoint_generation_},
    };
    const bool aggressor = captured_.front().launch_id == "aggressor";
    json victims = json::array();
    for (const auto& launch : captured_) {
      if (launch.launch_id != "aggressor") victims.push_back(launch.launch_id);
    }
    json matcher = json::array();
    for (const auto& expected : expected_) {
      matcher.push_back({
          {"launch_id", expected.at("launch_id")},
          {"capture_matcher", expected.at("capture_matcher")},
          {"occurrence_ordinal",
           expected.at("frame_local_launch_occurrence_ordinal")},
          {"sequence_index", expected.at("frame_local_launch_sequence_index")},
          {"sequence_anchors", expected.at("sequence_anchors")},
      });
    }
    json fragment = {
        {"schema", "kernel_capsule_v3"},
        {"capsule_id", sha256(specification_.dump())},
        {"source_run_id", specification_.value("source_run_id", "unknown")},
        {"aggressor_launch_id", aggressor ? json("aggressor") : json(nullptr)},
        {"victim_launch_ids", victims},
        {"launches", launches},
        {"dependencies", json::array()},
        {"clients", json::array({client})},
        {"capture_complete", true},
        {"applications_blocked", true},
        {"recursive_callbacks_disabled", true},
        {"no_later_model_work_submitted", true},
        {"victim_set", "gpu_execution_overlap_all_streams"},
        {"overlap_set_complete", specification_.at("overlap_set_complete")},
        {"overlap_evidence", specification_.at("overlap_evidence")},
        {"quiescent_checkpoint", true},
        {"submission_gate_enforced", true},
        {"event_operations", json::array()},
        {"terminal_streams", streams},
        {"checkpoint_generation", checkpoint_generation_},
        {"capture_matcher_evidence", matcher},
        {"agent_build", {{"version", PPERF_AGENT_VERSION},
                          {"cuda", CUDA_VERSION},
                          {"completion", "terminal_stream_synchronize"}}},
        {"submission_policy", "common_epoch_dependency_burst"},
        {"client_release_offsets_ns",
         specification_.at("client_release_offsets_ns")},
    };
    for (std::size_t index = 1; index < captured_.size(); ++index) {
      if (captured_[index - 1].launch_id != "aggressor" &&
          captured_[index].launch_id != "aggressor" &&
          captured_[index - 1].stream == captured_[index].stream) {
        fragment["dependencies"].push_back({
            {"source_launch_id", captured_[index - 1].launch_id},
            {"target_launch_id", captured_[index].launch_id},
            {"kind", "same_stream"},
            {"event_handle", nullptr},
        });
      }
    }
    for (const auto& operation : event_operations_) {
      fragment["event_operations"].push_back(operation.manifest());
      fragment["dependencies"].push_back({
          {"source_launch_id", operation.source_launch_id},
          {"target_launch_id", operation.target_launch_id},
          {"kind", "event"},
          {"event_handle", pointer_string(operation.event)},
      });
    }
    return {{"capsule_fragment", fragment}};
  }

  json restore_rpc() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.require(pperf::AgentState::frozen, "restore checkpoint");
    AgentCudaScope bypass;
    require_cuda(cuCtxPushCurrent(checkpoint_context_), "cuCtxPushCurrent");
    const CUptiResult result = pperf_checkpoint_restore(checkpoint_.get());
    CUcontext popped = nullptr;
    require_cuda(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
    require_cupti(result, "cuptiCheckpointRestore");
    return {{"restored", true},
            {"checkpoint_generation", checkpoint_generation_}};
  }

  json aggressor_priority_range_rpc() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.require(pperf::AgentState::frozen, "aggressor priority range");
    const auto aggressor = std::find_if(
        captured_.begin(), captured_.end(), [](const CapturedLaunch& launch) {
          return launch.launch_id == "aggressor";
        });
    if (aggressor == captured_.end()) {
      throw std::runtime_error("aggressor launch absent from owning client");
    }
    if (aggressor->api != "cuLaunchKernelEx") {
      throw std::runtime_error(
          "aggressor priority counterfactual requires cuLaunchKernelEx");
    }
    AgentCudaScope bypass;
    require_cuda(cuCtxPushCurrent(checkpoint_context_),
                 "priority range context push");
    int least_priority = 0;
    int greatest_priority = 0;
    require_cuda(cuCtxGetStreamPriorityRange(
        &least_priority, &greatest_priority), "cuCtxGetStreamPriorityRange");
    CUcontext popped = nullptr;
    require_cuda(cuCtxPopCurrent(&popped), "priority range context pop");
    int captured_priority = aggressor->stream_priority;
    for (const auto& attribute : aggressor->driver_attributes) {
      if (attribute.id == CU_LAUNCH_ATTRIBUTE_PRIORITY) {
        captured_priority = attribute.value.priority;
      }
    }
    return {
        {"supported", least_priority != greatest_priority},
        {"least_priority", least_priority},
        {"greatest_priority", greatest_priority},
        {"captured_priority", captured_priority},
        {"aggressor_launch_api", aggressor->api},
    };
  }

  json prepare_rpc(const json& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.require(pperf::AgentState::frozen, "prepare replay");
    replay_variant_ = request.at("variant");
    replay_instrumentation_ = request.at("instrumentation");
    replay_aggressor_priority_.reset();
    if (request.value("submission_policy", "") !=
        "common_epoch_dependency_burst") {
      throw std::runtime_error(
          "capsule_invalid: replay submission policy changed");
    }
    if (replay_variant_ != "pair" && replay_variant_ != "victim_only" &&
        replay_variant_ != "aggressor_only") {
      throw std::runtime_error("capsule_invalid: unknown replay variant");
    }
    if (replay_instrumentation_ != "none" &&
        replay_instrumentation_ != "nvbit_cta" &&
        replay_instrumentation_ != "launch_priority_counterfactual") {
      throw std::runtime_error(
          "optional replay collector unavailable for this agent build");
    }
    if (replay_instrumentation_ == "launch_priority_counterfactual") {
      if (replay_variant_ != "pair" ||
          !request.contains("aggressor_priority") ||
          request.at("aggressor_priority").is_null()) {
        throw std::runtime_error(
            "priority counterfactual requires pair and aggressor priority");
      }
      replay_aggressor_priority_ =
          request.at("aggressor_priority").get<int>();
    }
    if (replay_instrumentation_ == "nvbit_cta" &&
        replay_variant_ != "pair") {
      throw std::runtime_error("NVBit CTA collection requires pair replay");
    }
    AgentCudaScope bypass;
    require_cuda(cuCtxPushCurrent(checkpoint_context_),
                 "validation context push");
    for (const auto& launch : captured_) {
      CUmodule module = nullptr;
      require_cuda(cuFuncGetModule(&module, launch.function),
                   "validate cuFuncGetModule");
      const auto module_hash = module_hashes_.find(module);
      if (module_hash == module_hashes_.end() ||
          module_hash->second != launch.code_sha) {
        throw std::runtime_error(
            "capsule_invalid: replay module hash changed");
      }
      for (const auto& item : launch.function_attributes) {
        int observed = 0;
        require_cuda(cuFuncGetAttribute(
            &observed, static_cast<CUfunction_attribute>(
                std::stoi(item.first)), launch.function),
            "validate cuFuncGetAttribute");
        if (observed != item.second) {
          throw std::runtime_error(
              "capsule_invalid: replay function attributes changed");
        }
      }
      if (sha256(launch.command_payload().dump()) !=
          launch.captured_driver_fingerprint) {
        throw std::runtime_error(
            "capsule_invalid: Driver-command fingerprint changed");
      }
      for (const auto& parameter : launch.parameters) {
        if (!parameter.allocation_id) continue;
        const auto allocation = std::find_if(
            allocations_.begin(), allocations_.end(), [&](const auto& item) {
              return item.second.id == *parameter.allocation_id;
            });
        if (allocation == allocations_.end() ||
            !parameter.allocation_offset) {
          throw std::runtime_error(
              "capsule_invalid: parameter allocation identity changed");
        }
        std::uint64_t pointer = 0;
        std::memcpy(&pointer, parameter.bytes.data(), sizeof(pointer));
        if (pointer != allocation->second.base +
                           *parameter.allocation_offset) {
          throw std::runtime_error(
              "capsule_invalid: parameter allocation offset changed");
        }
      }
    }
    nvbit_cta_runtime_error_.clear();
    nvbit_cta_launch_slots_.clear();
    nvbit_cta_tracked_launches_.clear();
    nvbit_cta_clock_offset_ns_ = 0;
    nvbit_cta_clock_error_ns_ = 0;
    if (replay_instrumentation_ == "nvbit_cta") {
      if (!nvbit_cta_.available()) {
        nvbit_cta_runtime_error_ = nvbit_cta_.reason;
      } else {
        std::vector<pperf_nvbit_cta_launch_spec> specifications;
        for (auto& launch : captured_) {
          if ((replay_variant_ == "victim_only" &&
               launch.launch_id == "aggressor") ||
              (replay_variant_ == "aggressor_only" &&
               launch.launch_id != "aggressor")) continue;
          const auto slot = static_cast<std::uint32_t>(specifications.size());
          specifications.push_back({
              reinterpret_cast<std::uint64_t>(launch.function), slot,
              launch.grid[0], launch.grid[1], launch.grid[2],
              launch.block[0], launch.block[1], launch.block[2]});
          nvbit_cta_launch_slots_[launch.launch_id] = slot;
          nvbit_cta_tracked_launches_.push_back(&launch);
        }
        const int prepared = nvbit_cta_.prepare(
            specifications.data(), specifications.size());
        if (prepared != 0) {
          nvbit_cta_runtime_error_ =
              "NVBit CTA prepare failed: " + std::to_string(prepared);
        } else {
          std::uint64_t before = 0;
          std::uint64_t after = 0;
          std::uint64_t globaltimer = 0;
          require_cupti(cuptiGetTimestamp(&before),
                        "CTA calibration start timestamp");
          const int calibrated = nvbit_cta_.calibrate(&globaltimer);
          require_cupti(cuptiGetTimestamp(&after),
                        "CTA calibration end timestamp");
          if (calibrated != 0 || after < before) {
            nvbit_cta_runtime_error_ =
                "NVBit CTA calibration failed: " +
                std::to_string(calibrated);
          } else {
            const auto midpoint = before + (after - before) / 2;
            const auto difference = midpoint >= globaltimer ?
                midpoint - globaltimer : globaltimer - midpoint;
            if (difference > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
              nvbit_cta_runtime_error_ =
                  "NVBit CTA calibration offset overflow";
            } else {
              nvbit_cta_clock_offset_ns_ = midpoint >= globaltimer ?
                  static_cast<std::int64_t>(difference) :
                  -static_cast<std::int64_t>(difference);
              nvbit_cta_clock_error_ns_ = (after - before + 1) / 2;
            }
          }
        }
      }
    }
    CUcontext popped = nullptr;
    require_cuda(cuCtxPopCurrent(&popped), "validation context pop");
    state_.transition(pperf::AgentState::replay_prepared);
    return {{"prepared", true}};
  }

  void issue(CapturedLaunch& launch) {
    if (replay_instrumentation_ == "nvbit_cta" &&
        nvbit_cta_runtime_error_.empty()) {
      const auto slot = nvbit_cta_launch_slots_.find(launch.launch_id);
      if (slot == nvbit_cta_launch_slots_.end()) {
        nvbit_cta_runtime_error_ = "NVBit CTA launch slot missing";
      } else {
        const int identified = nvbit_cta_.identify(slot->second);
        if (identified != 0) {
          nvbit_cta_runtime_error_ =
              "NVBit CTA launch identification failed: " +
              std::to_string(identified);
        }
      }
    }
    replay_launch_ = &launch;
    void** parameters = launch.parameter_pointers.empty() ? nullptr :
        launch.parameter_pointers.data();
    std::size_t packed_size = launch.packed_parameters.size();
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER,
        launch.packed_parameters.empty() ? nullptr :
            launch.packed_parameters.data(),
        CU_LAUNCH_PARAM_BUFFER_SIZE,
        &packed_size,
        CU_LAUNCH_PARAM_END,
    };
    void** replay_parameters = launch.uses_packed_parameters ?
        nullptr : parameters;
    void** replay_extra = launch.uses_packed_parameters ? extra : nullptr;
    if (launch.api == "cuLaunchKernel") {
      require_cuda(cuLaunchKernel(
          launch.function, launch.grid[0], launch.grid[1], launch.grid[2],
          launch.block[0], launch.block[1], launch.block[2],
          launch.shared_memory, launch.stream, replay_parameters,
          replay_extra),
          "replay cuLaunchKernel");
    } else if (launch.api == "cuLaunchCooperativeKernel") {
      require_cuda(cuLaunchCooperativeKernel(
          launch.function, launch.grid[0], launch.grid[1], launch.grid[2],
          launch.block[0], launch.block[1], launch.block[2],
          launch.shared_memory, launch.stream, replay_parameters),
          "replay cuLaunchCooperativeKernel");
    } else if (launch.api == "cuLaunchKernelEx") {
      std::vector<CUlaunchAttribute> attributes = launch.driver_attributes;
      if (launch.launch_id == "aggressor" &&
          replay_aggressor_priority_) {
        auto priority = std::find_if(
            attributes.begin(), attributes.end(),
            [](const CUlaunchAttribute& attribute) {
              return attribute.id == CU_LAUNCH_ATTRIBUTE_PRIORITY;
            });
        if (priority == attributes.end()) {
          CUlaunchAttribute attribute{};
          attribute.id = CU_LAUNCH_ATTRIBUTE_PRIORITY;
          attribute.value.priority = *replay_aggressor_priority_;
          attributes.push_back(attribute);
        } else {
          priority->value.priority = *replay_aggressor_priority_;
        }
      }
      CUlaunchConfig config{
          launch.grid[0], launch.grid[1], launch.grid[2],
          launch.block[0], launch.block[1], launch.block[2],
          launch.shared_memory, launch.stream,
          attributes.data(),
          static_cast<unsigned int>(attributes.size())};
      require_cuda(cuLaunchKernelEx(&config, launch.function,
                                    replay_parameters, replay_extra),
                   "replay cuLaunchKernelEx");
    } else {
      throw std::runtime_error("capsule_invalid: unsupported replay API");
    }
    replay_launch_ = nullptr;
  }

  std::pair<std::string, std::size_t> digest_terminal_outputs() {
    if (!registered_outputs_.empty()) {
      SHA256_CTX hash;
      SHA256_Init(&hash);
      std::size_t copied = 0;
      for (const auto& output : registered_outputs_) {
        std::vector<std::uint8_t> host(output.size);
        require_cuda(cuMemcpyDtoH(host.data(), output.pointer, output.size),
                     "registered output validation copy");
        SHA256_Update(&hash, output.name.data(), output.name.size());
        SHA256_Update(&hash, host.data(), host.size());
        copied += host.size();
      }
      unsigned char digest[SHA256_DIGEST_LENGTH];
      SHA256_Final(digest, &hash);
      return {hex_bytes(digest, sizeof(digest)), copied};
    }
    std::set<std::string> ids;
    for (const auto& parameter : captured_.back().parameters) {
      if (parameter.allocation_id) ids.insert(*parameter.allocation_id);
    }
    SHA256_CTX hash;
    SHA256_Init(&hash);
    std::size_t copied = 0;
    for (const auto& item : allocations_) {
      if (!ids.count(item.second.id)) continue;
      std::vector<std::uint8_t> host(item.second.size);
      require_cuda(cuMemcpyDtoH(host.data(), item.second.base, item.second.size),
                   "terminal output validation copy");
      SHA256_Update(&hash, host.data(), host.size());
      copied += host.size();
    }
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &hash);
    return {hex_bytes(digest, sizeof(digest)), copied};
  }

  json execute_rpc(const json& request) {
    std::unique_lock<std::mutex> lock(mutex_);
    state_.require(pperf::AgentState::replay_prepared, "execute replay");
    state_.transition(pperf::AgentState::replaying);
    const auto release = request.at(
        "coordinator_release_ns").get<std::uint64_t>();
    const std::string variant = replay_variant_;
    const auto release_offset = request.at("client_release_offsets_ns")
        .at(client_id_).get<std::int64_t>();
    const auto client_release = static_cast<std::int64_t>(release) +
        release_offset;
    if (client_release <= 0) {
      throw std::runtime_error("capsule_invalid: client release underflow");
    }
    lock.unlock();
    sleep_until_monotonic(static_cast<std::uint64_t>(client_release));
    AgentCudaScope bypass;
    require_cuda(cuCtxPushCurrent(checkpoint_context_), "replay context push");
    std::uint64_t replay_epoch = 0;
    require_cupti(cuptiGetTimestamp(&replay_epoch), "replay epoch timestamp");
    std::vector<std::string> order;
    json fingerprints = json::object();
    for (auto& launch : captured_) {
      if ((variant == "victim_only" && launch.launch_id == "aggressor") ||
          (variant == "aggressor_only" &&
           launch.launch_id != "aggressor")) continue;
      for (const auto& operation : event_operations_) {
        if (operation.target_launch_id == launch.launch_id) {
          require_cuda(cuStreamWaitEvent(
              operation.wait_stream, operation.event,
              operation.wait_flags),
              "replay cuStreamWaitEvent");
        }
      }
      launch.replay_correlation = 0;
      require_cupti(cuptiGetTimestamp(&launch.driver_issue_start_ns),
                    "Driver issue start timestamp");
      issue(launch);
      require_cupti(cuptiGetTimestamp(&launch.driver_issue_end_ns),
                    "Driver issue end timestamp");
      for (const auto& operation : event_operations_) {
        if (operation.source_launch_id == launch.launch_id) {
          if (operation.record_api == "cuEventRecordWithFlags") {
            require_cuda(cuEventRecordWithFlags(
                operation.event, operation.record_stream,
                operation.record_flags),
                "replay cuEventRecordWithFlags");
          } else {
            require_cuda(cuEventRecord(
                operation.event, operation.record_stream),
                "replay cuEventRecord");
          }
        }
      }
      order.push_back(launch.launch_id);
      fingerprints[launch.launch_id] = launch.fingerprint();
    }
    for (CUstream stream : terminal_streams_) {
      require_cuda(cuStreamSynchronize(stream), "replay stream synchronize");
    }
    require_cupti(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
                  "replay activity flush");
    auto output = digest_terminal_outputs();
    CUcontext popped = nullptr;
    require_cuda(cuCtxPopCurrent(&popped), "replay context pop");
    std::optional<std::uint64_t> victim_start;
    std::optional<std::uint64_t> aggressor_duration;
    json launch_activities = json::array();
    {
      std::lock_guard<std::mutex> activity_lock(activity_mutex_);
      for (const auto& launch : captured_) {
        if ((variant == "victim_only" &&
             launch.launch_id == "aggressor") ||
            (variant == "aggressor_only" &&
             launch.launch_id != "aggressor")) {
          continue;
        }
        auto timing = activity_.find(launch.replay_correlation);
        if (timing == activity_.end()) continue;
        if (launch.launch_id == "aggressor") {
          aggressor_duration = timing->second.end - timing->second.start;
        } else if (!victim_start || timing->second.start < *victim_start) {
          victim_start = timing->second.start;
        }
        launch_activities.push_back({
            {"launch_id", launch.launch_id},
            {"client_id", launch.client_id},
            {"context_handle", pointer_string(launch.context)},
            {"stream_handle", pointer_string(launch.stream)},
            {"submission_ordinal", launch.submission_sequence_index},
            {"driver_command_fingerprint", launch.fingerprint()},
            {"driver_issue_start_ns", launch.driver_issue_start_ns},
            {"driver_issue_end_ns", launch.driver_issue_end_ns},
            {"gpu_start_ns", timing->second.start},
            {"gpu_end_ns", timing->second.end},
            {"gpu_duration_ns", timing->second.end - timing->second.start},
            {"issue_to_gpu_start_wait_ns",
             static_cast<std::int64_t>(timing->second.start) -
                 static_cast<std::int64_t>(launch.driver_issue_end_ns)},
            {"replay_ready_ns", launch.driver_issue_end_ns},
            {"admission_wait_ns",
             static_cast<std::int64_t>(timing->second.start) -
                 static_cast<std::int64_t>(launch.driver_issue_end_ns)},
        });
      }
    }
    lock.lock();
    state_.transition(pperf::AgentState::frozen);
    lock.unlock();
    json cta_intervals = json::array();
    json cta_collection_status = json::array();
    json clock_calibrations = json::array();
    bool cta_collection_valid = replay_instrumentation_ != "nvbit_cta";
    if (replay_instrumentation_ == "nvbit_cta") {
      std::size_t expected = 0;
      for (const auto* launch : nvbit_cta_tracked_launches_) {
        expected += static_cast<std::size_t>(launch->grid[0]) *
                    launch->grid[1] * launch->grid[2];
      }
      std::vector<pperf_nvbit_cta_record> observations(expected);
      std::size_t observed = 0;
      const int collected = nvbit_cta_.available() ? nvbit_cta_.collect(
          observations.data(), observations.size(), &observed) : 1;
      if (collected != 0 && nvbit_cta_runtime_error_.empty()) {
        nvbit_cta_runtime_error_ =
            "NVBit CTA collect failed: " + std::to_string(collected);
      }
      if (nvbit_cta_runtime_error_.empty() && observed == expected) {
        struct Counts {
          std::uint64_t expected{};
          std::uint64_t entered{};
          std::uint64_t exited{};
          std::uint64_t complete{};
        };
        std::vector<Counts> counts(nvbit_cta_tracked_launches_.size());
        for (std::size_t slot = 0; slot < counts.size(); ++slot) {
          const auto* launch = nvbit_cta_tracked_launches_[slot];
          counts[slot].expected =
              static_cast<std::uint64_t>(launch->grid[0]) *
              launch->grid[1] * launch->grid[2];
        }
        for (const auto& observation : observations) {
          if (observation.launch_slot >=
              nvbit_cta_tracked_launches_.size()) {
            nvbit_cta_runtime_error_ = "NVBit CTA returned invalid slot";
            break;
          }
          const auto* launch =
              nvbit_cta_tracked_launches_[observation.launch_slot];
          auto& count = counts[observation.launch_slot];
          const bool entered = observation.observation_flags &
              PPERF_NVBIT_CTA_ENTRY_OBSERVED;
          const bool exited = observation.observation_flags &
              PPERF_NVBIT_CTA_EXIT_OBSERVED;
          std::optional<std::uint64_t> entry;
          std::optional<std::uint64_t> exit;
          std::uint64_t converted = 0;
          if (entered && pperf::calibrated_timestamp(
                  observation.entry_globaltimer_ns,
                  nvbit_cta_clock_offset_ns_, converted)) entry = converted;
          if (exited && pperf::calibrated_timestamp(
                  observation.exit_globaltimer_ns,
                  nvbit_cta_clock_offset_ns_, converted)) exit = converted;
          if ((entered && !entry) || (exited && !exit) ||
              (entry && exit && *entry > *exit)) {
            nvbit_cta_runtime_error_ =
                "NVBit CTA returned invalid calibrated interval";
            break;
          }
          count.entered += entered;
          count.exited += exited;
          count.complete += entered && exited;
          cta_intervals.push_back({
              {"launch_id", launch->launch_id},
              {"client_id", client_id_},
              {"cta_id", observation.cta_id},
              {"sm_id", observation.sm_id ==
                      std::numeric_limits<std::uint32_t>::max() ?
                  json(nullptr) : json(observation.sm_id)},
              {"entry_ns", entry ? json(*entry) : json(nullptr)},
              {"exit_ns", exit ? json(*exit) : json(nullptr)},
              {"clock_error_ns", nvbit_cta_clock_error_ns_},
              {"entry_observed", entered},
              {"exit_observed", exited},
              {"observation_status", entered && exited ? "complete" :
                  entered ? "entry_only" : exited ? "exit_only" :
                  "missing"},
          });
        }
        if (nvbit_cta_runtime_error_.empty()) {
          for (std::size_t slot = 0; slot < counts.size(); ++slot) {
            const auto& count = counts[slot];
            cta_collection_status.push_back({
                {"launch_id", nvbit_cta_tracked_launches_[slot]->launch_id},
                {"client_id", client_id_},
                {"expected_count", count.expected},
                {"entered_count", count.entered},
                {"exited_count", count.exited},
                {"complete_count", count.complete},
                {"missing_count", count.expected - count.complete},
                {"dropped_entry_count", count.expected - count.entered},
                {"dropped_exit_count", count.expected - count.exited},
                {"coverage_fraction", count.expected == 0 ? 0.0 :
                    static_cast<double>(count.complete) / count.expected},
                {"quality", count.complete == count.expected ?
                    "complete" : "partial"},
            });
          }
          clock_calibrations.push_back({
              {"client_id", client_id_},
              {"clock_a", "globaltimer"},
              {"clock_b", "cupti_activity"},
              {"offset_ns", nvbit_cta_clock_offset_ns_},
              {"error_ns", nvbit_cta_clock_error_ns_},
              {"precision_class", "bounded_profiler_calibration"},
          });
          cta_collection_valid = true;
        }
      } else if (nvbit_cta_runtime_error_.empty()) {
        nvbit_cta_runtime_error_ = "NVBit CTA record count mismatch";
      }
      if (!cta_collection_valid) {
        cta_intervals = json::array();
        cta_collection_status = json::array();
        clock_calibrations = json::array();
      }
    }
    json result = {
        {"client_id", client_id_},
        {"victim_order", victim_ids()},
        {"launch_order", order},
        {"launch_fingerprints", fingerprints},
        {"fingerprint_match", true},
        {"victim_delay_ns", victim_start ?
             json(*victim_start - replay_epoch) : json(nullptr)},
        {"aggressor_duration_ns", aggressor_duration ?
             json(*aggressor_duration) : json(nullptr)},
        {"checkpoint_state_sha256", checkpoint_state_digest()},
        {"allocation_map_sha256", allocation_map_digest()},
        {"output_sha256", output.first},
        {"output_bytes_copied", output.second},
        {"foreign_work_count", foreign_work_count_},
        {"non_capsule_cuda_work_count", 0},
        {"frame_replay_count", 0},
        {"model_forward_count", 0},
        {"rosbag_work_count", 0},
        {"cta_intervals", cta_intervals},
        {"cta_collection_status", cta_collection_status},
        {"clock_calibrations", clock_calibrations},
        {"cta_collection_valid", cta_collection_valid},
        {"cta_collection_error", nvbit_cta_runtime_error_.empty() ?
             json(nullptr) : json(nvbit_cta_runtime_error_)},
        {"metric_samples", json::array()},
        {"launch_activities", launch_activities},
        {"source_gpu_offsets_used_for_submission", false},
        {"priority_attribute_applied", (
             replay_aggressor_priority_.has_value() &&
             std::any_of(captured_.begin(), captured_.end(),
                 [](const CapturedLaunch& launch) {
                   return launch.launch_id == "aggressor";
                 }))},
        {"aggressor_priority", replay_aggressor_priority_ ?
             json(*replay_aggressor_priority_) : json(nullptr)},
    };
    return result;
  }

  json victim_ids() const {
    json result = json::array();
    for (const auto& launch : captured_) {
      if (launch.launch_id != "aggressor") result.push_back(launch.launch_id);
    }
    return result;
  }

  std::string allocation_map_digest() const {
    json values = json::array();
    for (const auto& item : allocations_) {
      values.push_back({item.second.id, item.second.base,
                        item.second.size, item.second.kind});
    }
    return sha256(values.dump());
  }

  std::string checkpoint_state_digest() const {
    return sha256(client_id_ + ":" + std::to_string(checkpoint_generation_) +
                  ":" + allocation_map_digest());
  }

  json resource_rpc(const json& request) {
    const auto launch_id = request.at("launch_id").get<std::string>();
    auto found = std::find_if(captured_.begin(), captured_.end(),
        [&](const CapturedLaunch& launch) { return launch.launch_id == launch_id; });
    if (found == captured_.end()) {
      throw std::runtime_error("capsule_invalid: resource launch absent");
    }
    AgentCudaScope bypass;
    require_cuda(cuCtxPushCurrent(checkpoint_context_),
                 "resource context push");
    int blocks = 0;
    require_cuda(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, found->function,
        found->block[0] * found->block[1] * found->block[2],
        found->shared_memory), "occupancy query");
    CUcontext popped = nullptr;
    require_cuda(cuCtxPopCurrent(&popped), "resource context pop");
    return {
        {"function_attributes", found->function_attributes},
        {"grid_x", found->grid[0]},
        {"grid_y", found->grid[1]},
        {"grid_z", found->grid[2]},
        {"total_cta_count", found->grid[0] * found->grid[1] *
             found->grid[2]},
        {"block_x", found->block[0]},
        {"block_y", found->block[1]},
        {"block_z", found->block[2]},
        {"registers_per_thread",
         found->function_attributes.at(
             std::to_string(CU_FUNC_ATTRIBUTE_NUM_REGS))},
        {"static_shared_memory",
         found->function_attributes.at(
             std::to_string(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES))},
        {"dynamic_shared_memory", found->shared_memory},
        {"threads_per_block",
         found->block[0] * found->block[1] * found->block[2]},
        {"warps_per_block",
         (found->block[0] * found->block[1] * found->block[2] + 31) / 32},
        {"occupancy_blocks_per_sm", blocks},
        {"cupti_pm", {{"available", false},
          {"reason", "startup compatibility probe not enabled"}}},
        {"ncu_mps", {{"available", false},
          {"reason", "requires process relaunch under ncu --mps"}}},
        {"nvbit_cta", {{"available", nvbit_cta_.available()},
          {"version", nvbit_cta_.available() ?
              json(nvbit_cta_.version) : json(nullptr)},
          {"reason", nvbit_cta_.available() ?
              json(nullptr) : json(nvbit_cta_.reason)}}},
    };
  }

  json dispatch(const json& request) {
    update_lease(request);
    const auto operation = request.at("operation").get<std::string>();
    if (operation == "capabilities") return capabilities();
    if (operation == "lease_heartbeat") return {{"acknowledged", true}};
    if (operation == "capture_one_instrumented_frame") {
      return capture_rpc(request);
    }
    if (operation == "restore_checkpoint") return restore_rpc();
    if (operation == "aggressor_priority_range") {
      return aggressor_priority_range_rpc();
    }
    if (operation == "prepare_replay") return prepare_rpc(request);
    if (operation == "execute_replay") return execute_rpc(request);
    if (operation == "collect_resource_profile") return resource_rpc(request);
    if (operation == "shutdown") {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_.state() != pperf::AgentState::shutdown) {
        state_.transition(pperf::AgentState::shutdown);
      }
      shutdown_requested_ = true;
      condition_.notify_all();
      return {{"shutdown", true}};
    }
    throw std::runtime_error("unknown native agent operation");
  }

  void server_loop() noexcept {
    int server = -1;
    try {
      server = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
      if (server < 0) throw std::runtime_error("agent socket creation failed");
      sockaddr_un address{};
      address.sun_family = AF_UNIX;
      if (socket_path_.size() >= sizeof(address.sun_path)) {
        throw std::runtime_error("agent socket path is too long");
      }
      std::strncpy(address.sun_path, socket_path_.c_str(),
                   sizeof(address.sun_path) - 1);
      unlink(address.sun_path);
      if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
          0 || listen(server, 8) != 0) {
        throw std::runtime_error("agent socket bind/listen failed");
      }
      chmod(address.sun_path, 0600);
      while (!shutdown_requested_) {
        int connection = accept4(server, nullptr, nullptr, SOCK_CLOEXEC);
        if (connection < 0) {
          if (errno == EINTR) continue;
          throw std::runtime_error("agent socket accept failed");
        }
        std::string input;
        char buffer[65536];
        for (;;) {
          const auto count = read(connection, buffer, sizeof(buffer));
          if (count <= 0) break;
          input.append(buffer, static_cast<std::size_t>(count));
          if (input.find('\n') != std::string::npos) break;
          if (input.size() > 16 * 1024 * 1024) {
            throw std::runtime_error("agent request exceeds limit");
          }
        }
        json response;
        try {
          const json request = json::parse(input);
          if (request.value("protocol", "") != "kernel_capsule_agent_v3" ||
              request.value("client_id", "") != client_id_) {
            throw std::runtime_error("native agent protocol/client mismatch");
          }
          response = {{"protocol", "kernel_capsule_agent_v3"},
                      {"status", "ok"},
                      {"result", dispatch(request)}};
        } catch (const std::exception& error) {
          response = {{"protocol", "kernel_capsule_agent_v3"},
                      {"status", "error"},
                      {"error", std::string("capsule_invalid: ") + error.what()}};
        }
        const std::string output = response.dump() + "\n";
        (void)write(connection, output.data(), output.size());
        close(connection);
      }
      close(server);
      unlink(socket_path_.c_str());
      _Exit(0);
    } catch (const std::exception& error) {
      if (server >= 0) close(server);
      std::lock_guard<std::mutex> lock(mutex_);
      fail_locked(std::string("capsule_invalid: ") + error.what());
    }
  }

  void lease_loop() noexcept {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      std::string lease;
      std::uint64_t expiry = 0;
      pperf::AgentState state;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        lease = lease_id_;
        expiry = lease_expiry_ns_;
        state = state_.state();
      }
      if (state == pperf::AgentState::shutdown) return;
      if (!lease.empty() && monotonic_ns() > expiry) {
        const std::string abort_path = socket_path_ + ".abort.json";
        std::ofstream output(abort_path);
        output << json({{"schema", "kernel_capsule_abort_v3"},
                        {"client_id", client_id_},
                        {"reason", "coordinator_lease_expired"},
                        {"lease_id", lease}}).dump(2) << '\n';
        output.close();
        _Exit(86);
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  pperf::StateMachine state_;
  bool initialized_{false};
  std::atomic<bool> shutdown_requested_{false};
  std::string client_id_;
  std::string socket_path_;
  std::string specification_path_;
  json specification_;
  std::vector<json> expected_;
  std::vector<CapturedLaunch> captured_;
  std::vector<CapturedEventOperation> event_operations_;
  std::map<CUevent, std::size_t> recorded_events_;
  std::map<CUstream, std::vector<std::size_t>> pending_event_waits_;
  std::map<std::string, int> matcher_occurrences_;
  int frame_sequence_index_{-1};
  std::uint32_t active_selected_correlation_{0};
  std::string frame_model_;
  std::string frame_input_;
  std::uint64_t frame_stamp_{0};
  std::uint64_t capture_release_ns_{0};
  std::uint64_t capture_epoch_cupti_ns_{0};
  std::string error_;
  std::string lease_id_;
  std::uint64_t lease_expiry_ns_{0};
  CUpti_SubscriberHandle subscriber_{};
  std::thread server_thread_;
  std::thread lease_thread_;
  std::map<CUdeviceptr, Allocation> allocations_;
  std::vector<RegisteredOutput> registered_outputs_;
  std::uint64_t allocation_generation_{0};
  std::map<CUmodule, std::string> module_hashes_;
  std::map<CUfunction, CUmodule> function_modules_;
  std::map<CUstream, int> stream_priorities_;
  std::map<int, CUstream> source_stream_handles_;
  std::map<CUcontext, std::set<CUstream>> known_streams_;
  std::map<const void*, CUfunction> runtime_target_functions_;
  std::set<CUstream> terminal_streams_;
  struct CheckpointDeleter {
    void operator()(pperf_checkpoint* value) const {
      pperf_checkpoint_destroy(value);
    }
  };
  std::unique_ptr<pperf_checkpoint, CheckpointDeleter> checkpoint_;
  CUcontext checkpoint_context_{};
  int checkpoint_generation_{0};
  std::string replay_variant_;
  std::string replay_instrumentation_;
  std::optional<int> replay_aggressor_priority_;
  NvbitCtaApi nvbit_cta_;
  std::map<std::string, std::uint32_t> nvbit_cta_launch_slots_;
  std::vector<CapturedLaunch*> nvbit_cta_tracked_launches_;
  std::int64_t nvbit_cta_clock_offset_ns_{};
  std::uint64_t nvbit_cta_clock_error_ns_{};
  std::string nvbit_cta_runtime_error_;
  CapturedLaunch* replay_launch_{nullptr};
  std::uint64_t foreign_work_count_{0};
  std::mutex activity_mutex_;
  std::unordered_map<std::uint32_t, ActivityTiming> activity_;
  std::unordered_map<std::uint32_t, std::string> module_id_hashes_;
  std::map<std::string, std::set<std::string>> symbol_code_hashes_;
};

}  // namespace

struct pperf_checkpoint {
  CUpti_Checkpoint handle{};
};

extern "C" __attribute__((visibility("default")))
pperf_checkpoint* pperf_checkpoint_create(CUcontext context) {
  pperf_checkpoint* checkpoint = new (std::nothrow) pperf_checkpoint;
  if (checkpoint == nullptr) return nullptr;
  checkpoint->handle.structSize = CUpti_Checkpoint_STRUCT_SIZE;
  checkpoint->handle.ctx = context;
  checkpoint->handle.reserveDeviceMB = 0;
  checkpoint->handle.reserveHostMB = 0;
  checkpoint->handle.allowOverwrite = 0;
  checkpoint->handle.optimizations = 0;
  checkpoint->handle.pPriv = nullptr;
  return checkpoint;
}

extern "C" __attribute__((visibility("default")))
CUptiResult pperf_checkpoint_save(pperf_checkpoint* checkpoint) {
  if (checkpoint == nullptr) return CUPTI_ERROR_INVALID_PARAMETER;
  return NV::Cupti::Checkpoint::cuptiCheckpointSave(&checkpoint->handle);
}

extern "C" __attribute__((visibility("default")))
CUptiResult pperf_checkpoint_restore(pperf_checkpoint* checkpoint) {
  if (checkpoint == nullptr) return CUPTI_ERROR_INVALID_PARAMETER;
  return NV::Cupti::Checkpoint::cuptiCheckpointRestore(&checkpoint->handle);
}

extern "C" __attribute__((visibility("default")))
void pperf_checkpoint_destroy(pperf_checkpoint* checkpoint) {
  if (checkpoint == nullptr) return;
  if (checkpoint->handle.pPriv != nullptr) {
    NV::Cupti::Checkpoint::cuptiCheckpointFree(&checkpoint->handle);
  }
  delete checkpoint;
}

extern "C" __attribute__((visibility("default")))
CUresult pperf_capture_kernel_parameters(CUfunction function,
                                         void** kernel_parameters,
                                         pperf_parameter_sink sink,
                                         void* user_data) {
#if CUDA_VERSION >= 12040
  if (function == nullptr || kernel_parameters == nullptr || sink == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (std::size_t index = 0;; ++index) {
    std::size_t offset = 0;
    std::size_t size = 0;
    CUresult result = cuFuncGetParamInfo(function, index, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) return CUDA_SUCCESS;
    if (result != CUDA_SUCCESS) return result;
    if (kernel_parameters[index] == nullptr && size != 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    std::vector<std::uint8_t> bytes(size);
    if (size != 0) std::memcpy(bytes.data(), kernel_parameters[index], size);
    if (sink(index, offset, size, bytes.data(), user_data) != 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
#else
  (void)function; (void)kernel_parameters; (void)sink; (void)user_data;
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

extern "C" __attribute__((visibility("default")))
CUresult pperf_pointer_allocation(CUdeviceptr pointer, CUdeviceptr* base,
                                  std::size_t* allocation_size,
                                  std::size_t* allocation_offset) {
  if (base == nullptr || allocation_size == nullptr ||
      allocation_offset == nullptr) return CUDA_ERROR_INVALID_VALUE;
  CUresult result = cuMemGetAddressRange(base, allocation_size, pointer);
  if (result != CUDA_SUCCESS) return result;
  *allocation_offset = static_cast<std::size_t>(pointer - *base);
  return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("default")))
int InitializeInjection(void) {
  return KernelCapsuleAgent::instance().initialize();
}

extern "C" __attribute__((visibility("default")))
int pperf_agent_report_warmup_complete(const char* model_id) {
  return KernelCapsuleAgent::instance().report_warmup(model_id);
}

extern "C" __attribute__((visibility("default")))
int pperf_agent_wait_capture_epoch(std::uint64_t timeout_ms) {
  return KernelCapsuleAgent::instance().wait_capture_epoch(timeout_ms);
}

extern "C" __attribute__((visibility("default")))
int pperf_agent_set_frame(const char* model_id, const char* input_id,
                          std::uint64_t ros_header_timestamp_ns) {
  return KernelCapsuleAgent::instance().set_frame(
      model_id, input_id, ros_header_timestamp_ns);
}

extern "C" __attribute__((visibility("default")))
int pperf_agent_push_owner(const char* owner) {
  return KernelCapsuleAgent::instance().push_owner(owner);
}

extern "C" __attribute__((visibility("default")))
int pperf_agent_pop_owner(void) {
  return KernelCapsuleAgent::instance().pop_owner();
}

extern "C" __attribute__((visibility("default")))
int pperf_agent_report_frame_end(int success) {
  return KernelCapsuleAgent::instance().frame_end(success);
}

extern "C" __attribute__((visibility("default")))
int pperf_agent_register_output(CUdeviceptr pointer, std::size_t size,
                                const char* name) {
  return KernelCapsuleAgent::instance().register_output(pointer, size, name);
}
