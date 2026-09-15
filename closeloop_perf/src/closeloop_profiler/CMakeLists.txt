cmake_minimum_required(VERSION 3.22)
project(closeloop_profiler LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json 3.2 REQUIRED)

ament_python_install_package(${PROJECT_NAME})

find_path(CUPTI_INCLUDE_DIR cupti_checkpoint.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
find_library(CUPTI_LIBRARY cupti HINTS ${CUDAToolkit_LIBRARY_DIR})
find_library(CHECKPOINT_LIBRARY checkpoint HINTS ${CUDAToolkit_LIBRARY_DIR})
if(NOT CUPTI_INCLUDE_DIR OR NOT CUPTI_LIBRARY OR NOT CHECKPOINT_LIBRARY)
  message(FATAL_ERROR "CUPTI activity/callback/checkpoint libraries required")
endif()

add_library(pperf_kernel_capsule_agent SHARED
  native/agent_state.cpp
  native/kernel_capsule_native.cpp
)
target_compile_features(pperf_kernel_capsule_agent PRIVATE cxx_std_17)
target_compile_definitions(pperf_kernel_capsule_agent PRIVATE PPERF_AGENT_VERSION="3.0.0")
target_include_directories(pperf_kernel_capsule_agent PRIVATE ${CUPTI_INCLUDE_DIR})
target_link_libraries(pperf_kernel_capsule_agent PRIVATE
  CUDA::cuda_driver ${CUPTI_LIBRARY} ${CHECKPOINT_LIBRARY}
  OpenSSL::Crypto Threads::Threads nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS}
)
set_target_properties(pperf_kernel_capsule_agent PROPERTIES
  OUTPUT_NAME pperf_kernel_capsule_agent CXX_VISIBILITY_PRESET hidden)

set(NVBIT_ROOT "/opt/nvbit" CACHE PATH "NVBit release root")
if(EXISTS "${NVBIT_ROOT}/core/nvbit.h" AND EXISTS "${NVBIT_ROOT}/core/libnvbit.a")
  enable_language(CUDA)
  add_library(pperf_nvbit_cta_inject OBJECT native/nvbit_cta_inject.cu)
  target_include_directories(pperf_nvbit_cta_inject PRIVATE native ${NVBIT_ROOT}/core)
  target_compile_options(pperf_nvbit_cta_inject PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-astoolspatch>
    $<$<COMPILE_LANGUAGE:CUDA>:--keep-device-functions>
    $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=24>)
  set_target_properties(pperf_nvbit_cta_inject PROPERTIES
    CUDA_ARCHITECTURES 89 POSITION_INDEPENDENT_CODE ON)
  add_library(pperf_nvbit_cta_tracker SHARED
    native/nvbit_cta_tracker.cu $<TARGET_OBJECTS:pperf_nvbit_cta_inject>)
  target_include_directories(pperf_nvbit_cta_tracker PRIVATE native ${NVBIT_ROOT}/core)
  target_compile_options(pperf_nvbit_cta_tracker PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-cloning=no>)
  target_link_libraries(pperf_nvbit_cta_tracker PRIVATE
    ${NVBIT_ROOT}/core/libnvbit.a CUDA::cuda_driver CUDA::cudart_static
    Threads::Threads ${CMAKE_DL_LIBS})
  set_target_properties(pperf_nvbit_cta_tracker PROPERTIES
    OUTPUT_NAME pperf_nvbit_cta_tracker CUDA_ARCHITECTURES 89
    POSITION_INDEPENDENT_CODE ON)
  add_executable(pperf_nvbit_cta_validation native/nvbit_cta_validation.cu)
  target_include_directories(pperf_nvbit_cta_validation PRIVATE native)
  target_link_libraries(pperf_nvbit_cta_validation PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
  set_target_properties(pperf_nvbit_cta_validation PROPERTIES CUDA_ARCHITECTURES 89)
  install(TARGETS pperf_nvbit_cta_tracker pperf_nvbit_cta_validation
    LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()

install(TARGETS pperf_kernel_capsule_agent LIBRARY DESTINATION lib)
install(FILES
  native/agent_state.h native/kernel_capsule_native.h native/nvbit_cta_abi.h
  native/nvbit_cta_tracker_logic.h DESTINATION include/${PROJECT_NAME})

add_library(agent_state STATIC native/agent_state.cpp)
target_compile_features(agent_state PUBLIC cxx_std_17)
target_include_directories(agent_state PUBLIC native)
set_target_properties(agent_state PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_agent_state native/test/test_agent_state.cpp)
  target_link_libraries(test_agent_state agent_state)
  target_compile_features(test_agent_state PRIVATE cxx_std_17)
  ament_add_gtest(test_driver_replay native/test/test_driver_replay.cpp)
  target_include_directories(test_driver_replay PRIVATE ${CUPTI_INCLUDE_DIR})
  target_link_libraries(test_driver_replay CUDA::cuda_driver ${CUPTI_LIBRARY})
  target_compile_features(test_driver_replay PRIVATE cxx_std_17)
  ament_add_gtest(test_nvbit_cta_tracker native/test/test_nvbit_cta_tracker.cpp)
  target_include_directories(test_nvbit_cta_tracker PRIVATE native)
  target_compile_features(test_nvbit_cta_tracker PRIVATE cxx_std_17)
  ament_add_pytest_test(profiler_python_tests test)
endif()

ament_export_libraries(pperf_kernel_capsule_agent)
ament_export_include_directories(include/${PROJECT_NAME})
ament_package()
