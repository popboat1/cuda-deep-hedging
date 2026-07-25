#include "simulation.h"
#include <curand_kernel.h>

__global__ void setup_curand_kernel(curandState* states, unsigned long long seed, int num_paths) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_paths) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void gbm_path_kernel(curandState* states, float* d_paths, SimulationParams params) {
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(path_idx >= params.num_paths) return;

    // get RNG state from VRAM
    curandState local_state = states[path_idx];

    // pre-compute some constants
    float sigma_sq = params.sigma * params.sigma;
    float drift = (params.r - 0.5f * sigma_sq) * params.dt;
    float vol = params.sigma * sqrtf(params.dt);
    
    float curr_S = params.S0;
    
    int path_offset = path_idx * params.num_steps;
    
    for(int t = 0; t < params.num_steps; ++t){
        // write curr price to VRAM at step t
        d_paths[path_offset + t] = curr_S;
        // draw normal gaussian random noise
        float Z = curand_normal(&local_state);
        // advance stock price to t + 1
        curr_S *= expf(drift + vol * Z);
    }

    states[path_idx] = local_state;
}

__global__ void compute_payoffs(float* d_paths, float* d_payoffs, SimulationParams params){
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_idx >= params.num_paths) return;

    int final_step_idx = (path_idx * params.num_steps) + (params.num_steps - 1);

    float S_T = d_paths[final_step_idx];

    d_payoffs[path_idx] = fmaxf(S_T - params.K, 0);
}

void generate_gbm_paths(float* d_paths, float* d_payoffs, const SimulationParams& params, unsigned long long seed) {
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

    // compute payoffs
    compute_payoffs<<<blocks, threads>>>(d_paths, d_payoffs, params);
    CUDA_CHECK_KERNEL();
}