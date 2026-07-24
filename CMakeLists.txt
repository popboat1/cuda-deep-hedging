cmake_minimum_required(VERSION 3.18)
project(cuda_deep_hedging LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SOURCES 
    src/*.cpp 
    src/*.cu
)

add_executable(deep_hedger ${SOURCES})

target_link_libraries(deep_hedger PRIVATE CUDA::curand CUDA::cudart)

set_target_properties(deep_hedger PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)