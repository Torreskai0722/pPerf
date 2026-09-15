#include "nvbit_cta_abi.h"
#include "nvbit_cta_tracker_logic.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

TEST(NvbitCtaTracker, LinearizesThreeDimensionalCtas) {
  EXPECT_EQ(pperf::linear_cta_id(2, 3, 4, 7, 11), 2 + 7 * (3 + 11 * 4));
}

TEST(NvbitCtaTracker, DivergentExitsProduceOneCtaBoundary) {
  pperf::CtaAccumulator value;
  pperf::observe_entry(value, 20, 3);
  pperf::observe_entry(value, 10, 3);
  EXPECT_FALSE(pperf::observe_exit_group(value, 7, 32, 30));
  EXPECT_FALSE(pperf::observe_exit_group(value, 9, 32, 40));
  EXPECT_TRUE(pperf::observe_exit_group(value, 16, 32, 50));
  EXPECT_FALSE(pperf::observe_exit_group(value, 1, 32, 60));
  EXPECT_EQ(value.entry_ns, 10U);
  EXPECT_EQ(value.exit_ns, 50U);
  EXPECT_EQ(pperf::observation_flags(value),
            PPERF_NVBIT_CTA_ENTRY_OBSERVED |
                PPERF_NVBIT_CTA_EXIT_OBSERVED);
}

TEST(NvbitCtaTracker, ExitBoundaryAllowsWarpOvershoot) {
  pperf::CtaAccumulator value;
  value.exited_lanes = 496;
  EXPECT_TRUE(pperf::observe_exit_group(value, 32, 512, 50));
  EXPECT_EQ(value.exited_lanes, 528U);
  EXPECT_EQ(value.exit_ns, 50U);
  EXPECT_FALSE(pperf::observe_exit_group(value, 16, 512, 60));
  EXPECT_EQ(value.exit_ns, 50U);
}

TEST(NvbitCtaTracker, RetainsPartialBoundaries) {
  pperf::CtaAccumulator value;
  pperf::observe_entry(value, 10, 2);
  EXPECT_EQ(pperf::observation_flags(value),
            PPERF_NVBIT_CTA_ENTRY_OBSERVED);
  EXPECT_EQ(value.exit_ns, 0U);
}

TEST(NvbitCtaTracker, EnforcesFixedBufferBounds) {
  EXPECT_LT(pperf::kNvbitCtaMaxLaunches,
            std::numeric_limits<std::uint32_t>::max());
  EXPECT_LT(pperf::kNvbitCtaMaxRecords,
            std::numeric_limits<std::uint32_t>::max());
}

TEST(NvbitCtaTracker, SelectsOnlyThePassiveSequenceWindow) {
  EXPECT_FALSE(pperf::passive_sequence_selected(2, 3, 7));
  EXPECT_TRUE(pperf::passive_sequence_selected(3, 3, 7));
  EXPECT_TRUE(pperf::passive_sequence_selected(7, 3, 7));
  EXPECT_FALSE(pperf::passive_sequence_selected(8, 3, 7));
}

TEST(NvbitCtaTracker, AlignsExactHeadThenKeepsTailWindow) {
  EXPECT_TRUE(pperf::mixed_alignment_selected(265, 265, 0, 0));
  EXPECT_FALSE(pperf::mixed_alignment_selected(266, 265, 0, 0));
  EXPECT_FALSE(pperf::mixed_alignment_selected(265, 265, 1, 0));
  EXPECT_TRUE(pperf::passive_sequence_selected(290, 265, 290));
}

TEST(NvbitCtaTracker, ConvertsGlobaltimerWithSignedOffset) {
  std::uint64_t result = 0;
  EXPECT_TRUE(pperf::calibrated_timestamp(100, 25, result));
  EXPECT_EQ(result, 125U);
  EXPECT_TRUE(pperf::calibrated_timestamp(100, -25, result));
  EXPECT_EQ(result, 75U);
  EXPECT_FALSE(pperf::calibrated_timestamp(10, -25, result));
}

TEST(NvbitCtaTracker, PublicRecordHasEntryAndExitSmFields) {
  EXPECT_EQ(sizeof(pperf_nvbit_cta_record), 48U);
}

TEST(NvbitCtaTracker, CollectionStatusAccountsForCapacityAndBoundaries) {
  pperf_nvbit_cta_collection_status status{};
  status.record_capacity = 8;
  status.record_high_water_mark = 8;
  status.expected_record_count = 12;
  status.reserved_record_count = 8;
  status.dropped_capacity_record_count = 4;
  status.incomplete_exit_count = 1;
  EXPECT_EQ(status.expected_record_count,
            status.reserved_record_count +
                status.dropped_capacity_record_count);
  EXPECT_LE(status.record_high_water_mark, status.record_capacity);
}
