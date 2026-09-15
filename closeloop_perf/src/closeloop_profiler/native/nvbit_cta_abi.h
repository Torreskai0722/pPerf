#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PPERF_NVBIT_CTA_ABI_VERSION 7U
#define PPERF_NVBIT_CTA_VERSION "2.15.0"

enum {
  PPERF_NVBIT_CTA_ENTRY_OBSERVED = 1U,
  PPERF_NVBIT_CTA_EXIT_OBSERVED = 2U,
};

typedef struct pperf_nvbit_cta_launch_spec {
  uint64_t function_handle;
  uint32_t launch_slot;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  uint32_t block_x;
  uint32_t block_y;
  uint32_t block_z;
} pperf_nvbit_cta_launch_spec;

typedef struct pperf_nvbit_cta_record {
  uint32_t launch_slot;
  uint32_t cta_id;
  uint32_t sm_id;
  uint32_t exit_sm_id;
  uint32_t observation_flags;
  uint32_t nsmid;
  uint64_t grid_id;
  uint64_t entry_globaltimer_ns;
  uint64_t exit_globaltimer_ns;
} pperf_nvbit_cta_record;

typedef struct pperf_nvbit_cta_collection_status {
  uint64_t record_capacity;
  uint64_t record_high_water_mark;
  uint64_t expected_record_count;
  uint64_t reserved_record_count;
  uint64_t dropped_capacity_record_count;
  uint64_t incomplete_entry_count;
  uint64_t incomplete_exit_count;
} pperf_nvbit_cta_collection_status;

typedef int (*pperf_nvbit_cta_probe_fn)(uint32_t abi_version,
                                        const char** tracker_version);
typedef int (*pperf_nvbit_cta_prepare_fn)(
    const pperf_nvbit_cta_launch_spec* launches, size_t launch_count);
typedef int (*pperf_nvbit_cta_identify_next_launch_fn)(uint32_t launch_slot);
typedef int (*pperf_nvbit_cta_calibrate_fn)(uint64_t* globaltimer_ns);
typedef int (*pperf_nvbit_cta_collect_fn)(pperf_nvbit_cta_record* records,
                                          size_t capacity,
                                          size_t* record_count);
typedef int (*pperf_nvbit_cta_collection_status_fn)(
    pperf_nvbit_cta_collection_status* status);

int pperf_nvbit_cta_probe(uint32_t abi_version,
                          const char** tracker_version);
int pperf_nvbit_cta_prepare(const pperf_nvbit_cta_launch_spec* launches,
                            size_t launch_count);
int pperf_nvbit_cta_identify_next_launch(uint32_t launch_slot);
int pperf_nvbit_cta_calibrate(uint64_t* globaltimer_ns);
int pperf_nvbit_cta_collect(pperf_nvbit_cta_record* records, size_t capacity,
                            size_t* record_count);
int pperf_nvbit_cta_get_collection_status(
    pperf_nvbit_cta_collection_status* status);
int pperf_nvbit_cta_configure_from_environment(void);
int pperf_nvbit_cta_input_begin(void);
int pperf_nvbit_cta_input_end(void);
int pperf_nvbit_cta_arm(const char* label);
int pperf_nvbit_cta_disarm(void);

#ifdef __cplusplus
}
#endif
