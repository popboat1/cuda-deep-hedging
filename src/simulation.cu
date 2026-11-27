#include "simulation.cuh"
#include <curand_kernel.h>
#include <cmath>
#include <vector>
#include <numeric>

__global__ void setup_curand_kernel(curandState* states, unsigned long long seed, int num_paths) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_paths) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void compute_bs_delta(
    const float* d_paths,
    const float* d_vols,
    float* d_deltas,
    SimulationParams params
) {
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_idx >= params.num_paths) return;

    for (int t = 0; t < params.num_steps; ++t) {
        int curr_idx = t * params.num_paths + path_idx;
        float tau = params.T - (t * params.dt);
        float S_t = d_paths[curr_idx];

        if (tau <= 0.0f || t == params.num_steps - 1) {
            d_deltas[curr_idx] = (S_t > params.K) ? 1.0f : 0.0f;
        } else {
            // use instantaneous spot volatility so Heston benchmark tracks current variance
            float vol_t = fmaxf(d_vols[curr_idx], 1e-4f);
            float vol_sq = vol_t * vol_t;
            float d1 = (logf(S_t / params.K) + (params.r + 0.5f * vol_sq) * tau) / (vol_t * sqrtf(tau));
            d_deltas[curr_idx] = normcdff(d1);
        }
    }
}

__global__ void gbm_path_kernel(
    curandState* states,
    float* d_paths,
    float* d_vols,
    SimulationParams params
) {
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_idx >= params.num_paths) return;

    curandState local_state = states[path_idx];

    float sigma_sq = params.sigma * params.sigma;
    float drift = (params.r - 0.5f * sigma_sq) * params.dt;
    float vol = params.sigma * sqrtf(params.dt);
    float curr_S = params.S0;

    for (int t = 0; t < params.num_steps; ++t) {
        int curr_idx = t * params.num_paths + path_idx;
        d_paths[curr_idx] = curr_S;
        d_vols[curr_idx] = params.sigma;

        if (t < params.num_steps - 1) {
            float Z = curand_normal(&local_state);
            curr_S *= expf(drift + vol * Z);
        }
    }

    states[path_idx] = local_state;
}

__global__ void heston_path_kernel(
    curandState* states,
    float* d_paths,
    float* d_vols,
    SimulationParams params
) {
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_idx >= params.num_paths) return;

    curandState local_state = states[path_idx];

    int substeps = (params.heston_substeps > 0) ? params.heston_substeps : 8;
    float sub_dt = params.dt / static_cast<float>(substeps);
    float sqrt_sub_dt = sqrtf(sub_dt);
    float rho_comp = sqrtf(fmaxf(0.0f, 1.0f - params.rho * params.rho));

    float curr_S = params.S0;
    float curr_v = params.v0;

    for (int t = 0; t < params.num_steps; ++t) {
        int curr_idx = t * params.num_paths + path_idx;
        d_paths[curr_idx] = curr_S;
        d_vols[curr_idx] = sqrtf(fmaxf(curr_v, 0.0f));

        if (t < params.num_steps - 1) {
            // sub-stepped full truncation euler keeps variance discretization accurate
            for (int s = 0; s < substeps; ++s) {
                float z1 = curand_normal(&local_state);
                float z2 = curand_normal(&local_state);
                float dw1 = z1;
                float dw2 = params.rho * z1 + rho_comp * z2;

                float v_pos = fmaxf(curr_v, 0.0f);
                float sqrt_v = sqrtf(v_pos);

                curr_S *= expf((params.r - 0.5f * v_pos) * sub_dt + sqrt_v * sqrt_sub_dt * dw1);
                curr_v = curr_v + params.kappa * (params.theta - v_pos) * sub_dt
                         + params.xi * sqrt_v * sqrt_sub_dt * dw2;
            }
        }
    }

    states[path_idx] = local_state;
}

__global__ void compute_payoffs(const float* d_paths, float* d_payoffs, SimulationParams params) {
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_idx >= params.num_paths) return;

    int final_step_idx = (params.num_steps - 1) * params.num_paths + path_idx;
    float S_T = d_paths[final_step_idx];
    d_payoffs[path_idx] = fmaxf(S_T - params.K, 0.0f);
}

void generate_market_paths(
    float* d_paths,
    float* d_vols,
    float* d_payoffs,
    float* d_deltas,
    const SimulationParams& params,
    unsigned long long seed
) {
    int threads = 256;
    int blocks = cuda_utils::ceil_div(params.num_paths, threads);

    auto d_rng_states = cuda_utils::make_device_buffer<curandState>(params.num_paths);

    setup_curand_kernel<<<blocks, threads>>>(d_rng_states.get(), seed, params.num_paths);
    CUDA_CHECK_KERNEL();

    if (params.model == MarketModel::Heston) {
        heston_path_kernel<<<blocks, threads>>>(d_rng_states.get(), d_paths, d_vols, params);
    } else {
        gbm_path_kernel<<<blocks, threads>>>(d_rng_states.get(), d_paths, d_vols, params);
    }
    CUDA_CHECK_KERNEL();

    compute_payoffs<<<blocks, threads>>>(d_paths, d_payoffs, params);
    CUDA_CHECK_KERNEL();

    if (d_deltas != nullptr) {
        compute_bs_delta<<<blocks, threads>>>(d_paths, d_vols, d_deltas, params);
        CUDA_CHECK_KERNEL();
    }
}

float estimate_mc_option_price(
    const SimulationParams& params,
    int calibration_paths,
    unsigned long long seed
) {
    SimulationParams cal_params = params;
    cal_params.num_paths = calibration_paths;

    size_t total_elements = static_cast<size_t>(calibration_paths) * static_cast<size_t>(params.num_steps);
    auto d_paths = cuda_utils::make_device_buffer<float>(total_elements);
    auto d_vols = cuda_utils::make_device_buffer<float>(total_elements);
    auto d_payoffs = cuda_utils::make_device_buffer<float>(calibration_paths);

    generate_market_paths(d_paths.get(), d_vols.get(), d_payoffs.get(), nullptr, cal_params, seed);

    std::vector<float> h_payoffs(calibration_paths);
    CUDA_CHECK(cudaMemcpy(h_payoffs.data(), d_payoffs.get(), calibration_paths * sizeof(float), cudaMemcpyDeviceToHost));

    double mean_payoff = std::accumulate(h_payoffs.begin(), h_payoffs.end(), 0.0) / static_cast<double>(calibration_paths);
    return static_cast<float>(std::exp(-static_cast<double>(params.r) * static_cast<double>(params.T)) * mean_payoff);
}
