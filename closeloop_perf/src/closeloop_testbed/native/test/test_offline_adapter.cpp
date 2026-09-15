#include "offline_workload_abi.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

class OfflineAdapterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGetCount(&device_count_) != CUDA_SUCCESS ||
        device_count_ == 0) {
      GTEST_SKIP() << "CUDA device unavailable";
    }
    ASSERT_EQ(cuDeviceGet(&device_, 0), CUDA_SUCCESS);
    ASSERT_EQ(cuCtxCreate(&context_, 0, device_), CUDA_SUCCESS);
    api_ = pperf_offline_workload_v1_entry();
    ASSERT_NE(api_, nullptr);
    ASSERT_EQ(api_->abi_version, PPERF_OFFLINE_WORKLOAD_ABI_VERSION);
    const void* image = nullptr;
    std::size_t size = 0;
    ASSERT_EQ(api_->module_image(&image, &size), 0);
    ASSERT_NE(image, nullptr);
    ASSERT_GT(size, 0U);
    ASSERT_EQ(cuModuleLoadData(&module_, image), CUDA_SUCCESS);
  }

  void TearDown() override {
    if (module_ != nullptr) EXPECT_EQ(cuModuleUnload(module_), CUDA_SUCCESS);
    if (context_ != nullptr) EXPECT_EQ(cuCtxDestroy(context_), CUDA_SUCCESS);
  }

  int device_count_{};
  CUdevice device_{};
  CUcontext context_{};
  CUmodule module_{};
  const pperf_offline_workload_v1* api_{};
};

TEST_F(OfflineAdapterTest, RejectsMalformedConfiguration) {
  void* state = nullptr;
  char error[256]{};
  EXPECT_NE(api_->create("{\"blocks\":0}", module_, &state, error,
                         sizeof(error)), 0);
  EXPECT_EQ(state, nullptr);
}

TEST_F(OfflineAdapterTest, SizesAndHashesDeterministicOutput) {
  void* state = nullptr;
  char error[256]{};
  ASSERT_EQ(api_->create(
      "{\"blocks\":2,\"threads\":32,\"iterations\":1000}", module_,
      &state, error, sizeof(error)), 0) << error;
  CUstream stream = nullptr;
  ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  pperf_offline_launch_v1 launch{};
  ASSERT_EQ(api_->launch_count(state), 1U);
  ASSERT_EQ(api_->launch(state, 0, &launch), 0);
  EXPECT_EQ(launch.grid[0], 2U);
  EXPECT_EQ(launch.block[0], 32U);
  pperf_offline_output_v1 output{};
  ASSERT_EQ(api_->output_count(state), 1U);
  ASSERT_EQ(api_->output(state, 0, &output), 0);
  EXPECT_EQ(output.size, 2U * 32U * sizeof(float));
  std::vector<unsigned char> first(output.size);
  std::vector<unsigned char> second(output.size);
  for (auto* destination : {&first, &second}) {
    ASSERT_EQ(api_->reset(state, stream), 0);
    ASSERT_EQ(cuLaunchKernel(
        launch.function, launch.grid[0], launch.grid[1], launch.grid[2],
        launch.block[0], launch.block[1], launch.block[2],
        launch.dynamic_shared_memory, stream, launch.kernel_parameters,
        nullptr), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
    ASSERT_EQ(cuMemcpyDtoH(
        destination->data(), output.pointer, output.size), CUDA_SUCCESS);
  }
  EXPECT_EQ(first, second);
  EXPECT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
  api_->destroy(state);
}
