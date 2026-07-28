#include "portfolio.cuh"
#include <cmath>

__device__ inline float norm_cdf_dev(float x) {
    return 0.5f * erfcf(-x * M_SQRT1_2);
}

__global__ void compute_portfolio(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_deltas,
    float* d_portfolio_values,
    const SimulationParams params
){
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(path_idx >= params.num_paths) return;

    // pre-compute contants
    float exp_r_dt = expf(params.r * params.dt);

    // inception (t = 0)
    float S0 = d_paths[0 * params.num_paths + path_idx];
    float d0 = d_deltas[0 * params.num_paths + path_idx];
    float fee_0 = params.cost_ratio * fabsf(d0) * S0;
    float C = params.option_price - (d0 * S0) - fee_0;
    float prev_delta = d0;

    for(int t = 1; t < params.num_steps; t++){
        float S_t = d_paths[t * params.num_paths + path_idx];
        float d_t = d_deltas[t * params.num_paths + path_idx];

        float pos_change = d_t - prev_delta;
        float fee_t = params.cost_ratio * fabsf(pos_change) * S_t;
        C = (C * exp_r_dt) - (pos_change * S_t) - fee_t;
        prev_delta = d_t;
    }

    // expiry settlement (t = T)
    float S_T = d_paths[(params.num_steps - 1) * params.num_paths + path_idx];
    float payoff = d_payoffs[path_idx];
    float C_final = C * exp_r_dt;
    float V_T = C_final + (prev_delta * S_T) - payoff;

    d_portfolio_values[path_idx] = V_T;
}

__global__ void compute_portfolio_trajectories(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_deltas,
    float* d_trajectories,
    SimulationParams params
) {
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_idx >= params.num_paths) return;

    float exp_r_dt = expf(params.r * params.dt);
    float sigma_sq = params.sigma * params.sigma;

    float cash = params.option_price;
    float prev_delta = 0.0f;

    for (int t = 0; t < params.num_steps; ++t) {
        int curr_idx = t * params.num_paths + path_idx;
        float S_t = d_paths[curr_idx];
        float delta_t = d_deltas[curr_idx];

        float d_delta = delta_t - prev_delta;
        float trade_cost = fabsf(d_delta) * S_t * params.cost_ratio;

        if (t > 0) {
            cash *= exp_r_dt;
        }
        cash -= (d_delta * S_t + trade_cost);

        float tau = params.T - (t * params.dt);
        float option_val = 0.0f;

        int traj_idx = path_idx * params.num_steps + t;

        if (t == params.num_steps - 1 || tau <= 0.0f) {
            float final_cash = cash * exp_r_dt;
            float liquidation_cost = fabsf(delta_t) * S_t * params.cost_ratio;
            float payoff = d_payoffs[path_idx];
            d_trajectories[traj_idx] = final_cash + (delta_t * S_t) - liquidation_cost - payoff;
        } else {
            float d1 = (logf(S_t / params.K) + (params.r + 0.5f * sigma_sq) * tau) / (params.sigma * sqrtf(tau));
            float d2 = d1 - params.sigma * sqrtf(tau);
            option_val = S_t * norm_cdf_dev(d1) - params.K * expf(-params.r * tau) * norm_cdf_dev(d2);

            // net mark-to-market portfolio value
            d_trajectories[traj_idx] = cash + (delta_t * S_t) - option_val;
        }

        prev_delta = delta_t;
    }
}

void evaluate_portfolio(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_deltas,
    float* d_portfolio_values,
    const SimulationParams& params
){
    int threads = 256;
    int blocks = cuda_utils::ceil_div(params.num_paths, threads);

    compute_portfolio<<<blocks, threads>>>(
        d_paths, 
        d_payoffs,
        d_deltas,
        d_portfolio_values,
        params
    );
}

void evaluate_portfolio_trajectories(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_deltas,
    float* d_trajectories,
    const SimulationParams& params
) {
    int threads = 256;
    int blocks = cuda_utils::ceil_div(params.num_paths, threads);
    compute_portfolio_trajectories<<<blocks, threads>>>(d_paths, d_payoffs, d_deltas, d_trajectories, params);
    CUDA_CHECK_KERNEL();
}