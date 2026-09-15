#pragma once

#include <cuda.h>
#include <cupti_result.h>
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pperf_checkpoint pperf_checkpoint;

typedef int (*pperf_parameter_sink)(std::size_t index, std::size_t offset,
                                    std::size_t size, const void* bytes,
                                    void* user_data);

// Save is intentionally separate so an injected callback can invoke it
// immediately before the first selected launch in this client context.
pperf_checkpoint* pperf_checkpoint_create(CUcontext context);
CUptiResult pperf_checkpoint_save(pperf_checkpoint* checkpoint);
CUptiResult pperf_checkpoint_restore(pperf_checkpoint* checkpoint);
void pperf_checkpoint_destroy(pperf_checkpoint* checkpoint);

// Copy the exact driver parameter layout. CUDA <12.4 returns
// CUDA_ERROR_NOT_SUPPORTED and is rejected by the Python capability gate.
CUresult pperf_capture_kernel_parameters(CUfunction function,
                                         void** kernel_parameters,
                                         pperf_parameter_sink sink,
                                         void* user_data);

// Resolve a direct device pointer to its original live allocation and offset.
CUresult pperf_pointer_allocation(CUdeviceptr pointer, CUdeviceptr* base,
                                  std::size_t* allocation_size,
                                  std::size_t* allocation_offset);

// CUDA calls this entry point when the library is loaded through
// CUDA_INJECTION64_PATH. It subscribes before application CUDA initialization.
int InitializeInjection(void);

// Application lifecycle and semantic-owner markers. Zero means success;
// non-zero means the strict state machine has rejected the operation.
int pperf_agent_report_warmup_complete(const char* model_id);
int pperf_agent_wait_capture_epoch(std::uint64_t timeout_ms);
int pperf_agent_set_frame(const char* model_id, const char* input_id,
                          std::uint64_t ros_header_timestamp_ns);
int pperf_agent_push_owner(const char* owner);
int pperf_agent_pop_owner(void);
int pperf_agent_report_frame_end(int success);
// Register deterministic adapter outputs before capture/checkpoint creation.
int pperf_agent_register_output(CUdeviceptr pointer, std::size_t size,
                                const char* name);

#ifdef __cplusplus
}
#endif
