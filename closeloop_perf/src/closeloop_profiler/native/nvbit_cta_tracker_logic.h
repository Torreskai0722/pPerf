#pragma once

#include "nvbit_cta_abi.h"

#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#define PPERF_CTA_HD __host__ __device__
#else
#define PPERF_CTA_HD
#endif

namespace pperf {

constexpr std::uint32_t kNvbitCtaMaxLaunches = 1024;
constexpr std::uint64_t kNvbitCtaInactiveLaunch = kNvbitCtaMaxLaunches;
// ponytail: fixed buffer avoids post-checkpoint allocation; raise this only
// when a real capsule exceeds the recorded ceiling.
constexpr std::uint32_t kNvbitCtaMaxRecords = 3000000;

PPERF_CTA_HD constexpr bool passive_sequence_selected(
    std::uint32_t sequence, std::uint32_t start, std::uint32_t end) {
  return sequence >= start && sequence <= end;
}

PPERF_CTA_HD constexpr bool mixed_alignment_selected(
    std::uint32_t sequence, std::uint32_t sequence_start,
    std::uint32_t occurrence, std::uint32_t target_occurrence) {
  return sequence == sequence_start && occurrence == target_occurrence;
}

PPERF_CTA_HD constexpr std::uint64_t linear_cta_id(
    std::uint32_t x, std::uint32_t y, std::uint32_t z,
    std::uint32_t grid_x, std::uint32_t grid_y) {
  return x + static_cast<std::uint64_t>(grid_x) *
                 (y + static_cast<std::uint64_t>(grid_y) * z);
}

struct CtaAccumulator {
  std::uint64_t entry_ns{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t exit_ns{};
  std::uint32_t sm_id{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t exited_lanes{};
};

PPERF_CTA_HD constexpr bool completes_exit_group(
    std::uint32_t prior_lanes, std::uint32_t participating_lanes,
    std::uint32_t threads_per_cta) {
  return participating_lanes != 0 && prior_lanes < threads_per_cta &&
         prior_lanes + participating_lanes >= threads_per_cta;
}

inline void observe_entry(CtaAccumulator& value, std::uint64_t timestamp_ns,
                          std::uint32_t sm_id) {
  if (timestamp_ns < value.entry_ns) value.entry_ns = timestamp_ns;
  if (value.sm_id == std::numeric_limits<std::uint32_t>::max()) {
    value.sm_id = sm_id;
  }
}

inline bool observe_exit_group(CtaAccumulator& value,
                               std::uint32_t participating_lanes,
                               std::uint32_t threads_per_cta,
                               std::uint64_t timestamp_ns) {
  if (participating_lanes == 0 || value.exited_lanes >= threads_per_cta) {
    return false;
  }
  const auto prior = value.exited_lanes;
  value.exited_lanes += participating_lanes;
  if (!completes_exit_group(
          prior, participating_lanes, threads_per_cta)) return false;
  value.exit_ns = timestamp_ns;
  return true;
}

inline bool calibrated_timestamp(std::uint64_t raw_ns, std::int64_t offset_ns,
                                 std::uint64_t& result_ns) {
  if (offset_ns < 0) {
    const auto magnitude = static_cast<std::uint64_t>(-(offset_ns + 1)) + 1;
    if (raw_ns < magnitude) return false;
    result_ns = raw_ns - magnitude;
    return true;
  }
  const auto offset = static_cast<std::uint64_t>(offset_ns);
  if (raw_ns > std::numeric_limits<std::uint64_t>::max() - offset) {
    return false;
  }
  result_ns = raw_ns + offset;
  return true;
}

inline std::uint32_t observation_flags(const CtaAccumulator& value) {
  return (value.entry_ns != std::numeric_limits<std::uint64_t>::max()
              ? PPERF_NVBIT_CTA_ENTRY_OBSERVED
              : 0U) |
         (value.exit_ns != 0 ? PPERF_NVBIT_CTA_EXIT_OBSERVED : 0U);
}

}  // namespace pperf

#undef PPERF_CTA_HD
