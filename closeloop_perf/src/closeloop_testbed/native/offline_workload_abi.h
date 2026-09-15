#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define PPERF_OFFLINE_WORKLOAD_ABI_VERSION 1U

typedef struct pperf_offline_launch_v1 {
  CUfunction function;
  const char* symbol;
  std::uint32_t grid[3];
  std::uint32_t block[3];
  std::uint32_t dynamic_shared_memory;
  void** kernel_parameters;
} pperf_offline_launch_v1;

typedef struct pperf_offline_output_v1 {
  const char* name;
  CUdeviceptr pointer;
  std::size_t size;
} pperf_offline_output_v1;

typedef struct pperf_offline_workload_v1 {
  std::uint32_t abi_version;
  const char* adapter_name;
  int (*module_image)(const void** image, std::size_t* size);
  int (*create)(const char* configuration_json, CUmodule module,
                void** state, char* error, std::size_t error_size);
  int (*reset)(void* state, CUstream stream);
  std::size_t (*launch_count)(void* state);
  int (*launch)(void* state, std::size_t index,
                pperf_offline_launch_v1* launch);
  std::size_t (*output_count)(void* state);
  int (*output)(void* state, std::size_t index,
                pperf_offline_output_v1* output);
  void (*destroy)(void* state);
} pperf_offline_workload_v1;

typedef const pperf_offline_workload_v1* (
    *pperf_offline_workload_v1_entry_fn)(void);

const pperf_offline_workload_v1* pperf_offline_workload_v1_entry(void);

#ifdef __cplusplus
}
#endif
