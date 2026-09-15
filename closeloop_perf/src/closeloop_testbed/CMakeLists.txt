cmake_minimum_required(VERSION 3.22)
project(closeloop_testbed LANGUAGES CXX CUDA)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(closeloop_profiler REQUIRED)
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json 3.2 REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/model_node scripts/fma_workload_node scripts/capsule_model_client
  scripts/relay_node scripts/replayer_node DESTINATION lib/${PROJECT_NAME})

find_path(CUPTI_INCLUDE_DIR cupti_checkpoint.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
find_library(CUPTI_LIBRARY cupti HINTS ${CUDAToolkit_LIBRARY_DIR})
if(NOT CUPTI_INCLUDE_DIR OR NOT CUPTI_LIBRARY)
  message(FATAL_ERROR "CUPTI activity library required")
endif()

add_library(pperf_offline_compute_adapter SHARED native/offline_compute_adapter.cpp)
target_compile_features(pperf_offline_compute_adapter PRIVATE cxx_std_17)
target_include_directories(pperf_offline_compute_adapter PRIVATE native)
target_link_libraries(pperf_offline_compute_adapter PRIVATE
  CUDA::cuda_driver nlohmann_json::nlohmann_json)
set_target_properties(pperf_offline_compute_adapter PROPERTIES
  OUTPUT_NAME pperf_offline_compute_adapter CXX_VISIBILITY_PRESET hidden)

add_executable(pperf_offline_worker native/offline_worker.cpp)
target_compile_features(pperf_offline_worker PRIVATE cxx_std_17)
target_include_directories(pperf_offline_worker PRIVATE
  ${CUPTI_INCLUDE_DIR} native ${closeloop_profiler_INCLUDE_DIRS})
target_link_libraries(pperf_offline_worker PRIVATE
  CUDA::cuda_driver ${CUPTI_LIBRARY} OpenSSL::Crypto
  nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})

add_executable(pperf_mps_stage_microbenchmark native/mps_stage_microbenchmark.cu)
target_link_libraries(pperf_mps_stage_microbenchmark PRIVATE CUDA::cudart)
set_target_properties(pperf_mps_stage_microbenchmark PROPERTIES CUDA_ARCHITECTURES 89)

install(TARGETS
  pperf_offline_compute_adapter pperf_offline_worker pperf_mps_stage_microbenchmark
  LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(FILES native/offline_workload_abi.h DESTINATION include/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_offline_adapter native/test/test_offline_adapter.cpp)
  target_include_directories(test_offline_adapter PRIVATE native)
  target_link_libraries(test_offline_adapter
    pperf_offline_compute_adapter CUDA::cuda_driver)
  target_compile_features(test_offline_adapter PRIVATE cxx_std_17)
  ament_add_pytest_test(testbed_python_tests test)
endif()

ament_package()
