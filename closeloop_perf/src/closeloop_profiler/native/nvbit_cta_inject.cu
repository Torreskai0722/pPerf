#include "nvbit_cta_tracker_logic.h"

#include <climits>
#include <cstdint>

#include "utils/utils.h"

struct DeviceCtaAccumulator {
  unsigned long long entry_ns;
  unsigned long long exit_ns;
  unsigned long long grid_id;
  unsigned int sm_id;
  unsigned int exit_sm_id;
  unsigned int exited_lanes;
  unsigned int nsmid;
};

extern "C" __device__ __noinline__ void pperf_cta_entry(
    std::uint64_t launch_slot, std::uint64_t records_address,
    std::uint64_t bases_address) {
  if (launch_slot >= pperf::kNvbitCtaMaxLaunches) return;
  const auto active = __activemask();
  const auto first_lane = __ffs(active) - 1;
  if (get_laneid() != first_lane) return;
  auto* records = reinterpret_cast<DeviceCtaAccumulator*>(records_address);
  const auto* bases = reinterpret_cast<const std::uint32_t*>(bases_address);
  const auto cta_id = pperf::linear_cta_id(
      blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y);
  auto& record = records[bases[launch_slot] + cta_id];
  std::uint64_t now = 0;
  std::uint64_t grid_id = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid_id));
  atomicMin(&record.entry_ns, static_cast<unsigned long long>(now));
  atomicCAS(&record.grid_id, ULLONG_MAX,
            static_cast<unsigned long long>(grid_id));
  atomicCAS(&record.sm_id, UINT32_MAX, get_smid());
  unsigned int nsmid = 0;
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsmid));
  atomicCAS(&record.nsmid, UINT32_MAX, nsmid);
}

extern "C" __device__ __noinline__ void pperf_cta_exit(
    int predicate, std::uint64_t launch_slot, std::uint64_t records_address,
    std::uint64_t bases_address, std::uint64_t threads_address) {
  if (launch_slot >= pperf::kNvbitCtaMaxLaunches) return;
  const auto active = __activemask();
  const auto participating = __ballot_sync(active, predicate);
  if (participating == 0) return;
  const auto first_lane = __ffs(participating) - 1;
  if (get_laneid() != first_lane) return;
  auto* records = reinterpret_cast<DeviceCtaAccumulator*>(records_address);
  const auto* bases = reinterpret_cast<const std::uint32_t*>(bases_address);
  const auto* threads = reinterpret_cast<const std::uint32_t*>(threads_address);
  const auto cta_id = pperf::linear_cta_id(
      blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y);
  auto& record = records[bases[launch_slot] + cta_id];
  const auto lanes = static_cast<unsigned int>(__popc(participating));
  const auto prior = atomicAdd(&record.exited_lanes, lanes);
  if (!pperf::completes_exit_group(
          prior, lanes, threads[launch_slot])) return;
  std::uint64_t now = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  atomicExch(&record.exit_sm_id, get_smid());
  atomicExch(&record.exit_ns, static_cast<unsigned long long>(now));
}
