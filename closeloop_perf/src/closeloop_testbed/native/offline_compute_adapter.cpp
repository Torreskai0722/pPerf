#include "offline_workload_abi.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <string>

namespace {

constexpr char kComputePtx[] = R"ptx(
.version 7.0
.target sm_52
.address_size 64

.visible .entry pperf_offline_fma(
    .param .u64 output,
    .param .u64 iterations
)
{
    .reg .pred %p;
    .reg .b32 %thread, %blockid, %blockdim, %index;
    .reg .b64 %output, %address, %iterations, %counter;
    .reg .f32 %value, %seed;
    ld.param.u64 %output, [output];
    ld.param.u64 %iterations, [iterations];
    mov.u32 %thread, %tid.x;
    mov.u32 %blockid, %ctaid.x;
    mov.u32 %blockdim, %ntid.x;
    mad.lo.u32 %index, %blockid, %blockdim, %thread;
    cvt.rn.f32.u32 %seed, %index;
    mul.f32 %seed, %seed, 0f358637BD;
    add.f32 %value, %seed, 0f3F000000;
    mov.u64 %counter, 0;
loop:
    fma.rn.f32 %value, %value, 0f3F7FBE77, %seed;
    add.u64 %counter, %counter, 1;
    setp.lt.u64 %p, %counter, %iterations;
    @%p bra loop;
    mul.wide.u32 %address, %index, 4;
    add.u64 %address, %output, %address;
    st.global.f32 [%address], %value;
    ret;
}
)ptx";

struct State {
  CUfunction function{};
  CUdeviceptr output{};
  std::uint64_t iterations{};
  std::uint32_t blocks{};
  std::uint32_t threads{};
  void* parameters[2]{};
};

void message(char* output, std::size_t size, const std::string& value) {
  if (output == nullptr || size == 0) return;
  std::snprintf(output, size, "%s", value.c_str());
}

int module_image(const void** image, std::size_t* size) {
  if (image == nullptr || size == nullptr) return 1;
  *image = kComputePtx;
  *size = sizeof(kComputePtx) - 1;
  return 0;
}

int create(const char* raw, CUmodule module, void** output_state,
           char* error, std::size_t error_size) {
  if (raw == nullptr || module == nullptr || output_state == nullptr) return 1;
  try {
    const auto config = nlohmann::json::parse(raw);
    const auto blocks = config.at("blocks").get<std::uint64_t>();
    const auto threads = config.at("threads").get<std::uint64_t>();
    const auto iterations = config.at("iterations").get<std::uint64_t>();
    if (blocks == 0 || blocks > std::numeric_limits<std::uint32_t>::max() ||
        threads == 0 || threads > 1024 || iterations == 0) {
      throw std::runtime_error("blocks, threads, and iterations must be positive");
    }
    auto* state = new State;
    state->blocks = static_cast<std::uint32_t>(blocks);
    state->threads = static_cast<std::uint32_t>(threads);
    state->iterations = iterations;
    CUresult result = cuModuleGetFunction(
        &state->function, module, "pperf_offline_fma");
    if (result != CUDA_SUCCESS) {
      delete state;
      throw std::runtime_error("pperf_offline_fma is absent from module");
    }
    const std::size_t bytes = state->blocks * state->threads * sizeof(float);
    result = cuMemAlloc(&state->output, bytes);
    if (result != CUDA_SUCCESS) {
      delete state;
      throw std::runtime_error("output allocation failed");
    }
    state->parameters[0] = &state->output;
    state->parameters[1] = &state->iterations;
    *output_state = state;
    return 0;
  } catch (const std::exception& exception) {
    message(error, error_size, exception.what());
    return 1;
  }
}

int reset(void* raw, CUstream stream) {
  if (raw == nullptr) return 1;
  auto* state = static_cast<State*>(raw);
  return cuMemsetD32Async(
      state->output, 0, state->blocks * state->threads, stream) == CUDA_SUCCESS
      ? 0 : 1;
}

std::size_t launch_count(void*) { return 1; }

int launch(void* raw, std::size_t index, pperf_offline_launch_v1* output) {
  if (raw == nullptr || output == nullptr || index != 0) return 1;
  auto* state = static_cast<State*>(raw);
  *output = {state->function, "pperf_offline_fma",
             {state->blocks, 1, 1}, {state->threads, 1, 1}, 0,
             state->parameters};
  return 0;
}

std::size_t output_count(void*) { return 1; }

int output(void* raw, std::size_t index, pperf_offline_output_v1* output) {
  if (raw == nullptr || output == nullptr || index != 0) return 1;
  auto* state = static_cast<State*>(raw);
  *output = {"result", state->output,
             state->blocks * state->threads * sizeof(float)};
  return 0;
}

void destroy(void* raw) {
  auto* state = static_cast<State*>(raw);
  if (state == nullptr) return;
  if (state->output != 0) cuMemFree(state->output);
  delete state;
}

const pperf_offline_workload_v1 kApi = {
    PPERF_OFFLINE_WORKLOAD_ABI_VERSION, "builtin_compute_v1",
    module_image, create, reset, launch_count, launch,
    output_count, output, destroy};

}  // namespace

extern "C" __attribute__((visibility("default")))
const pperf_offline_workload_v1* pperf_offline_workload_v1_entry() {
  return &kApi;
}
