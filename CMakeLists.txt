cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 86)
project(cuda_deep_hedging LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CORE_SOURCES
    src/autograd.cu
    src/dataloader.cpp
    src/exporter.cu
    src/mlp.cu
    src/optimizer.cu
    src/portfolio.cu
    src/simulation.cu
)

add_compile_options(
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)

add_executable(deep_hedger src/main.cpp ${CORE_SOURCES})
target_link_libraries(deep_hedger PRIVATE CUDA::curand CUDA::cudart)
set_target_properties(deep_hedger PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

enable_testing()
add_executable(test_hedger tests/test_hedger.cu ${CORE_SOURCES})
target_link_libraries(test_hedger PRIVATE CUDA::curand CUDA::cudart)
set_target_properties(test_hedger PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
add_test(NAME hedger_tests COMMAND test_hedger)
