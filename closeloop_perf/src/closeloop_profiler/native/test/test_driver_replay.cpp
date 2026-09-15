#include <cuda.h>
#include <cupti.h>
#include <cupti_activity.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct ActivityInterval {
  std::uint64_t start{};
  std::uint64_t end{};
};

std::vector<std::uint32_t> launch_correlations;
std::map<std::uint32_t, ActivityInterval> launch_activity;

void CUPTIAPI ApiCallback(void*, CUpti_CallbackDomain domain,
                          CUpti_CallbackId, const void* callback_data) {
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
  const auto* data = static_cast<const CUpti_CallbackData*>(callback_data);
  if (data->callbackSite != CUPTI_API_ENTER || data->functionName == nullptr) {
    return;
  }
  const std::string function(data->functionName);
  if (function.rfind("cuLaunchKernel", 0) == 0 ||
      function.rfind("cuLaunchCooperativeKernel", 0) == 0) {
    launch_correlations.push_back(data->correlationId);
  }
}

void CUPTIAPI BufferRequested(std::uint8_t** buffer, std::size_t* size,
                              std::size_t* max_records) {
  constexpr std::size_t kBufferSize = 1024 * 1024;
  void* storage = nullptr;
  if (posix_memalign(&storage, 8, kBufferSize) != 0) storage = nullptr;
  *buffer = static_cast<std::uint8_t*>(storage);
  *size = storage == nullptr ? 0 : kBufferSize;
  *max_records = 0;
}

void CUPTIAPI BufferCompleted(CUcontext, std::uint32_t, std::uint8_t* buffer,
                              std::size_t, std::size_t valid_size) {
  CUpti_Activity* record = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid_size, &record) ==
         CUPTI_SUCCESS) {
    if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL &&
        record->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
      continue;
    }
    const auto* kernel =
        reinterpret_cast<const CUpti_ActivityKernel9*>(record);
    launch_activity[kernel->correlationId] = {
        kernel->start, kernel->end,
    };
  }
  std::free(buffer);
}

constexpr char kPtx[] = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry add_one(
    .param .u64 output
)
{
    .reg .b32 %value;
    .reg .b64 %address;
    ld.param.u64 %address, [output];
    ld.global.u32 %value, [%address];
    add.u32 %value, %value, 1;
    st.global.u32 [%address], %value;
    ret;
}
)ptx";

class DriverReplayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGetCount(&device_count_) != CUDA_SUCCESS ||
        device_count_ == 0) {
      GTEST_SKIP() << "CUDA device unavailable";
    }
    ASSERT_EQ(cuDeviceGet(&device_, 0), CUDA_SUCCESS);
    ASSERT_EQ(cuCtxCreate(&context_, 0, device_), CUDA_SUCCESS);
    ASSERT_EQ(cuModuleLoadData(&module_, kPtx), CUDA_SUCCESS);
    ASSERT_EQ(cuModuleGetFunction(&function_, module_, "add_one"),
              CUDA_SUCCESS);
    ASSERT_EQ(cuMemAlloc(&output_, sizeof(std::uint32_t)), CUDA_SUCCESS);
    ASSERT_EQ(cuMemsetD32(output_, 0, 1), CUDA_SUCCESS);
    launch_correlations.clear();
    launch_activity.clear();
    ASSERT_EQ(cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted),
              CUPTI_SUCCESS);
    ASSERT_EQ(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
              CUPTI_SUCCESS);
    ASSERT_EQ(cuptiSubscribe(&subscriber_, ApiCallback, nullptr),
              CUPTI_SUCCESS);
    ASSERT_EQ(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API),
              CUPTI_SUCCESS);
  }

  void TearDown() override {
    if (subscriber_ != nullptr) {
      EXPECT_EQ(cuptiUnsubscribe(subscriber_), CUPTI_SUCCESS);
    }
    EXPECT_EQ(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
              CUPTI_SUCCESS);
    if (output_ != 0) EXPECT_EQ(cuMemFree(output_), CUDA_SUCCESS);
    if (module_ != nullptr) EXPECT_EQ(cuModuleUnload(module_), CUDA_SUCCESS);
    if (context_ != nullptr) EXPECT_EQ(cuCtxDestroy(context_), CUDA_SUCCESS);
  }

  void** parameters() {
    parameter_ = output_;
    parameter_pointer_ = &parameter_;
    return &parameter_pointer_;
  }

  int device_count_{};
  CUdevice device_{};
  CUcontext context_{};
  CUmodule module_{};
  CUfunction function_{};
  CUdeviceptr output_{};
  CUdeviceptr parameter_{};
  void* parameter_pointer_{};
  CUpti_SubscriberHandle subscriber_{};
};

TEST_F(DriverReplayTest, ReplaysAllDriverVariantsAndDependencies) {
  CUstream first = nullptr;
  CUstream second = nullptr;
  CUevent edge = nullptr;
  ASSERT_EQ(cuStreamCreate(&first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&second, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuEventCreate(&edge, CU_EVENT_DISABLE_TIMING), CUDA_SUCCESS);

  ASSERT_EQ(cuLaunchKernel(function_, 1, 1, 1, 1, 1, 1, 0, first,
                           parameters(), nullptr),
            CUDA_SUCCESS);

  CUdeviceptr packed_parameter = output_;
  std::size_t packed_size = sizeof(packed_parameter);
  void* extra[] = {
      CU_LAUNCH_PARAM_BUFFER_POINTER,
      &packed_parameter,
      CU_LAUNCH_PARAM_BUFFER_SIZE,
      &packed_size,
      CU_LAUNCH_PARAM_END,
  };
  ASSERT_EQ(cuLaunchKernel(function_, 1, 1, 1, 1, 1, 1, 0, first, nullptr,
                           extra),
            CUDA_SUCCESS);

  CUlaunchAttribute priority{};
  priority.id = CU_LAUNCH_ATTRIBUTE_PRIORITY;
  priority.value.priority = 0;
  CUlaunchConfig extended{
      1, 1, 1, 1, 1, 1, 0, first, &priority, 1,
  };
  ASSERT_EQ(cuLaunchKernelEx(&extended, function_, parameters(), nullptr),
            CUDA_SUCCESS);

  int cooperative = 0;
  ASSERT_EQ(cuDeviceGetAttribute(
                &cooperative, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
                device_),
            CUDA_SUCCESS);
  if (cooperative) {
    ASSERT_EQ(cuLaunchCooperativeKernel(
                  function_, 1, 1, 1, 1, 1, 1, 0, first, parameters()),
              CUDA_SUCCESS);
  }

  ASSERT_EQ(cuLaunchKernel(function_, 1, 1, 1, 1, 1, 1, 0, first,
                           parameters(), nullptr),
            CUDA_SUCCESS);
  ASSERT_EQ(cuEventRecord(edge, first), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamWaitEvent(second, edge, 0), CUDA_SUCCESS);
  ASSERT_EQ(cuLaunchKernel(function_, 1, 1, 1, 1, 1, 1, 0, second,
                           parameters(), nullptr),
            CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(second), CUDA_SUCCESS);
  ASSERT_EQ(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
            CUPTI_SUCCESS);

  std::uint32_t observed = 0;
  ASSERT_EQ(cuMemcpyDtoH(&observed, output_, sizeof(observed)), CUDA_SUCCESS);
  EXPECT_EQ(observed, cooperative ? 6u : 5u);
  const std::size_t expected_launches = cooperative ? 6u : 5u;
  ASSERT_EQ(launch_correlations.size(), expected_launches);
  for (std::uint32_t correlation : launch_correlations) {
    const auto found = launch_activity.find(correlation);
    ASSERT_NE(found, launch_activity.end())
        << "missing CUPTI activity for correlation " << correlation;
    EXPECT_LT(found->second.start, found->second.end);
  }

  EXPECT_EQ(cuEventDestroy(edge), CUDA_SUCCESS);
  EXPECT_EQ(cuStreamDestroy(second), CUDA_SUCCESS);
  EXPECT_EQ(cuStreamDestroy(first), CUDA_SUCCESS);
}

TEST_F(DriverReplayTest, AppliesPerLaunchPriorityRangeExtremes) {
  int least_priority = 0;
  int greatest_priority = 0;
  ASSERT_EQ(cuCtxGetStreamPriorityRange(
                &least_priority, &greatest_priority),
            CUDA_SUCCESS);
  if (least_priority == greatest_priority) {
    GTEST_SKIP() << "CUDA device exposes no launch-priority range";
  }
  CUstream stream = nullptr;
  ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  for (int requested : {greatest_priority, least_priority}) {
    CUlaunchAttribute priority{};
    priority.id = CU_LAUNCH_ATTRIBUTE_PRIORITY;
    priority.value.priority = requested;
    CUlaunchConfig config{
        1, 1, 1, 1, 1, 1, 0, stream, &priority, 1,
    };
    ASSERT_EQ(cuLaunchKernelEx(
                  &config, function_, parameters(), nullptr),
              CUDA_SUCCESS);
  }
  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  std::uint32_t observed = 0;
  ASSERT_EQ(cuMemcpyDtoH(&observed, output_, sizeof(observed)), CUDA_SUCCESS);
  EXPECT_EQ(observed, 2u);
  EXPECT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
}

}  // namespace
