#include "portfolio.h"

__global__ void compute_portfolio(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_deltas,
    float* d_portfolio_values,
    const SimulationParams params
){
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(path_idx >= params.num_paths) return;

    int path_offset = path_idx * params.num_steps;

    // pre-compute contants
    float exp_r_dt = expf(params.r * params.dt);

    // inception (t = 0)
    float S0 = d_paths[path_offset];
    float d0 = d_deltas[path_offset];
    float fee_0 = params.cost_ratio * fabsf(d0) * S0;
    float C = params.option_price - (d0 * S0) - fee_0;
    float prev_delta = d0;

    for(int t = 1; t < params.num_steps; t++){
        float S_t = d_paths[path_offset + t];
        float d_t = d_deltas[path_offset + t];

        float pos_change = d_t - prev_delta;
        float fee_t = params.cost_ratio * fabsf(pos_change) * S_t;
        C = (C * exp_r_dt) - (pos_change * S_t) - fee_t;
        prev_delta = d_t;
    }

    // expiry settlement (t = T)
    float S_T = d_paths[path_offset + params.num_steps - 1];
    float payoff = d_payoffs[path_idx];
    float C_final = C * exp_r_dt;
    float V_T = C_final + (prev_delta * S_T) - payoff;

    d_portfolio_values[path_idx] = V_T;
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