// Controlled kernels used only to validate the four offline MPS stage skills.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

__global__ void persistent_grid_kernel(std::uint64_t cycles,
                                       volatile unsigned int* entered) {
  atomicAdd(const_cast<unsigned int*>(entered), 1U);
  const std::uint64_t begin = clock64();
  while (clock64() - begin < cycles) {
    asm volatile("" ::: "memory");
  }
}

template <int RegisterWords>
__global__ void residency_kernel(std::uint64_t cycles, float* output) {
  extern __shared__ unsigned char dynamic_shared[];
  float values[RegisterWords];
#pragma unroll
  for (int index = 0; index < RegisterWords; ++index) {
    values[index] = static_cast<float>(threadIdx.x + index);
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dynamic_shared[0] = static_cast<unsigned char>(RegisterWords);
  }
  const std::uint64_t begin = clock64();
  while (clock64() - begin < cycles) {
    asm volatile("" ::: "memory");
  }
  if (threadIdx.x == 0) {
    output[blockIdx.x] = values[RegisterWords - 1];
  }
}

void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

std::string argument(int argc, char** argv, const std::string& name,
                     const std::string& fallback = "") {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) return argv[index + 1];
  }
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argument(argc, argv, "--mode", "persistent_grid");
    const std::string output_path = argument(argc, argv, "--output");
    if (output_path.empty()) throw std::runtime_error("--output is required");
    const int blocks = std::stoi(argument(argc, argv, "--blocks", "56"));
    const int threads = std::stoi(argument(argc, argv, "--threads", "256"));
    const int shared = std::stoi(argument(argc, argv, "--shared-bytes", "0"));
    const std::uint64_t cycles = std::stoull(
        argument(argc, argv, "--cycles", "10000000"));
    unsigned int* entered = nullptr;
    float* result = nullptr;
    check(cudaMalloc(&entered, sizeof(unsigned int)), "cudaMalloc entered");
    check(cudaMalloc(&result, sizeof(float) * blocks), "cudaMalloc result");
    check(cudaMemset(entered, 0, sizeof(unsigned int)), "cudaMemset entered");
    cudaEvent_t begin;
    cudaEvent_t end;
    check(cudaEventCreate(&begin), "cudaEventCreate begin");
    check(cudaEventCreate(&end), "cudaEventCreate end");
    check(cudaEventRecord(begin), "cudaEventRecord begin");
    if (mode == "residency") {
      residency_kernel<64><<<blocks, threads, shared>>>(cycles, result);
    } else {
      persistent_grid_kernel<<<blocks, threads>>>(cycles, entered);
    }
    check(cudaGetLastError(), "kernel launch");
    check(cudaEventRecord(end), "cudaEventRecord end");
    check(cudaEventSynchronize(end), "cudaEventSynchronize");
    float duration_ms = 0.0F;
    check(cudaEventElapsedTime(&duration_ms, begin, end),
          "cudaEventElapsedTime");
    unsigned int entered_count = 0;
    check(cudaMemcpy(&entered_count, entered, sizeof(unsigned int),
                     cudaMemcpyDeviceToHost), "cudaMemcpy entered");
    std::ofstream output(output_path);
    output << "{\n"
           << "  \"mode\": \"" << mode << "\",\n"
           << "  \"blocks\": " << blocks << ",\n"
           << "  \"threads\": " << threads << ",\n"
           << "  \"shared_bytes\": " << shared << ",\n"
           << "  \"duration_ms\": " << duration_ms << ",\n"
           << "  \"entered_blocks\": " << entered_count << ",\n"
           << "  \"direct_wdu_observation\": false\n"
           << "}\n";
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(entered);
    cudaFree(result);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
