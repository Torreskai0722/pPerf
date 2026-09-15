#include "nvbit_cta_abi.h"

#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

__global__ void validation_kernel_a(std::uint64_t* output,
                                    std::uint64_t iterations) {
  std::uint64_t value = blockIdx.x * blockDim.x + threadIdx.x + 1;
  for (std::uint64_t index = 0; index < iterations; ++index) {
    value = value * 2862933555777941757ULL + 3037000493ULL;
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

__global__ void validation_kernel_large(std::uint64_t* output,
                                        std::uint64_t iterations) {
  extern __shared__ std::uint64_t shared[];
  std::uint64_t value = blockIdx.x * blockDim.x + threadIdx.x + 11;
  for (std::uint64_t index = 0; index < iterations; ++index) {
    value = value * 2862933555777941757ULL + 3037000493ULL;
  }
  shared[threadIdx.x] = value;
  __syncthreads();
  output[blockIdx.x * blockDim.x + threadIdx.x] =
      value ^ shared[(threadIdx.x + 1) % blockDim.x];
}

__global__ void validation_kernel_b(std::uint64_t* output,
                                    std::uint64_t iterations) {
  std::uint64_t value = blockIdx.x * blockDim.x + threadIdx.x + 7;
  for (std::uint64_t index = 0; index < iterations; ++index) {
    value = value * 3202034522624059733ULL + 1ULL;
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

__global__ void validation_kernel_divergent(std::uint64_t* output) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if ((threadIdx.x & 1U) == 0) {
    output[index] = index;
    return;
  }
  std::uint64_t value = index;
  for (std::uint32_t iteration = 0; iteration < 1000; ++iteration) {
    value = value * 2862933555777941757ULL + 3037000493ULL;
  }
  output[index] = value;
}

void require(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

std::string argument(int argc, char** argv, const std::string& name,
                     const std::string& fallback = "") {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) return argv[index + 1];
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const std::string& name) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] == name) return true;
  }
  return false;
}

void barrier(const std::string& root, const std::string& client) {
  if (root.empty()) return;
  std::ofstream(root + "/" + client + ".ready").put('\n');
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream first(root + "/first.ready");
    std::ifstream second(root + "/second.ready");
    if (first.good() && second.good()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  throw std::runtime_error("two-client validation barrier timed out");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto test_case = argument(argc, argv, "--case", "known");
    const auto client = argument(argc, argv, "--client", "single");
    const auto barrier_root = argument(argc, argv, "--barrier");
    const auto no_tracer = has_flag(argc, argv, "--no-tracer");
    auto arm = reinterpret_cast<int (*)(const char*)>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_arm"));
    auto disarm = reinterpret_cast<int (*)()>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_disarm"));
    auto input_begin = reinterpret_cast<int (*)()>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_input_begin"));
    auto input_end = reinterpret_cast<int (*)()>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_input_end"));
    auto configure = reinterpret_cast<int (*)()>(
        dlsym(RTLD_DEFAULT, "pperf_nvbit_cta_configure_from_environment"));
    const auto* mode = std::getenv("PPERF_NVBIT_CTA_MODE");
    const bool passive = mode != nullptr && std::string(mode) == "passive";
    if (!no_tracer && (arm == nullptr || disarm == nullptr)) {
      throw std::runtime_error("NVBit CTA standalone ABI is unavailable");
    }
    if (!no_tracer && passive &&
        (input_begin == nullptr || input_end == nullptr ||
         configure == nullptr)) {
      throw std::runtime_error("NVBit CTA passive input ABI is unavailable");
    }
    if (!no_tracer && passive && configure() != 0) {
      throw std::runtime_error("failed to configure passive CTA tracker");
    }
    require(cudaFree(nullptr), "CUDA initialization");
    std::uint64_t* output = nullptr;
    require(cudaMalloc(&output, 112 * 256 * sizeof(*output)), "cudaMalloc");
    validation_kernel_b<<<1, 1>>>(output, 1);
    validation_kernel_a<<<1, 1>>>(output, 1);
    require(cudaDeviceSynchronize(), "warmup");
    if (!no_tracer && passive) {
      const auto begin_status = input_begin();
      if (begin_status != 0) {
        throw std::runtime_error(
            "failed to begin CTA learning input: " +
            std::to_string(begin_status));
      }
      if (test_case == "divergent") {
        validation_kernel_divergent<<<7, 64>>>(output);
      } else if (test_case == "multi_stream" ||
                 test_case == "multi_stream_long") {
        validation_kernel_a<<<5, 64>>>(output, 1);
        validation_kernel_b<<<9, 64>>>(output, 1);
      } else if (test_case == "resource_large") {
        require(cudaFuncSetAttribute(
            validation_kernel_large,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            65536), "learning large shared-memory opt-in");
        validation_kernel_large<<<112, 128, 65536>>>(output, 1);
      } else if (test_case == "resource_small") {
        validation_kernel_a<<<112, 128>>>(output, 1);
      } else if (test_case == "window_gate") {
        validation_kernel_a<<<5, 64>>>(output, 1);
        validation_kernel_a<<<6, 64>>>(output, 1);
        validation_kernel_a<<<7, 64>>>(output, 1);
      } else {
        validation_kernel_a<<<test_case == "buffer" ? 8 : 7, 64>>>(
            output, 1);
      }
      require(cudaDeviceSynchronize(), "learning input synchronize");
      const auto end_status = input_end();
      if (end_status != 0) {
        throw std::runtime_error(
            "failed to end CTA learning input: " +
            std::to_string(end_status));
      }
    }
    if (!no_tracer) {
      const auto arm_status = arm("synthetic");
      if (arm_status != 0) {
        throw std::runtime_error(
            "failed to arm CTA tracker: " + std::to_string(arm_status));
      }
    }
    if (test_case == "known") {
      validation_kernel_a<<<7, 64>>>(output, 1000);
    } else if (test_case == "divergent") {
      validation_kernel_divergent<<<7, 64>>>(output);
    } else if (test_case == "buffer") {
      validation_kernel_a<<<8, 64>>>(output, 1000);
    } else if (test_case == "window_gate") {
      validation_kernel_a<<<5, 64>>>(output, 1000);
      validation_kernel_a<<<6, 64>>>(output, 1000);
      validation_kernel_a<<<7, 64>>>(output, 1000);
    } else if (test_case == "multi_stream" ||
               test_case == "multi_stream_long") {
      const auto iterations = test_case == "multi_stream_long"
                                  ? 5000000ULL : 100000ULL;
      cudaStream_t first{};
      cudaStream_t second{};
      require(cudaStreamCreate(&first), "first stream");
      require(cudaStreamCreate(&second), "second stream");
      validation_kernel_a<<<5, 64, 0, first>>>(output, iterations);
      validation_kernel_b<<<9, 64, 0, second>>>(output, iterations);
      require(cudaStreamSynchronize(first), "first stream synchronize");
      require(cudaStreamSynchronize(second), "second stream synchronize");
      require(cudaStreamDestroy(first), "first stream destroy");
      require(cudaStreamDestroy(second), "second stream destroy");
    } else if (test_case == "resource_small") {
      barrier(barrier_root, client);
      validation_kernel_a<<<112, 128>>>(output, 300000);
    } else if (test_case == "resource_large") {
      require(cudaFuncSetAttribute(
          validation_kernel_large,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          65536), "large shared-memory opt-in");
      barrier(barrier_root, client);
      validation_kernel_large<<<112, 128, 65536>>>(output, 300000);
    } else {
      throw std::runtime_error("unknown validation case: " + test_case);
    }
    require(cudaDeviceSynchronize(), "validation synchronize");
    if (!no_tracer && disarm() != 0) {
      throw std::runtime_error("failed to disarm CTA tracker");
    }
    require(cudaFree(output), "cudaFree");
    std::cout << "{\"case\":\"" << test_case << "\",\"client\":\""
              << client << "\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
