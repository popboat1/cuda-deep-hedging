#include "simulation.h"
#include <curand_kernel.h>

__global__ void setup_curand_kernel(curandState* states, unsigned long seed, int num_paths) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_paths) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void gbm_path_kernel(curandState* states, float* d_paths, SimulationParams params) {
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(path_idx >= params.num_paths) return;

    // TODO: Implement Geometric Brownian Motion loop
}

void generate_gbm_paths(float* d_paths, const SimulationParams& params, unsigned long long seed) {
    int threads = 256;
    int blocks = cuda_utils::ceil_div(params.num_paths, threads);

    // alloc cuRAND states in VRAM
    auto d_rng_states = cuda_utils::make_device_buffer<curandState>(params.num_paths);

    // cuRAND init
    setup_curand_kernel<<<blocks, threads>>>(d_rng_states.get(), seed, params.num_paths);
    CUDA_CHECK_KERNEL();

    // GBM generator
    gbm_path_kernel<<<blocks, threads>>>(d_rng_states.get(), d_paths, params);
    CUDA_CHECK_KERNEL();
}