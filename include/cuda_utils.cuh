#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <memory>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

#define CUDA_CHECK_KERNEL() \
    do { \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

namespace cuda_utils{

template <typename T1, typename T2>
constexpr __host__ __device__ inline T1 ceil_div(T1 total, T2 block_size){
    return (total + block_size - 1) / block_size;
}

struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

template <typename T>
using device_ptr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
device_ptr<T> make_device_buffer(size_t count) {
    T* raw_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw_ptr), count * sizeof(T)));
    return device_ptr<T>(raw_ptr);
}

} // namespace cuda_utils

#endif