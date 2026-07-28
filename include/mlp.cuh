#ifndef MLP_H
#define MLP_H

#include "simulation.cuh"

constexpr int input_dim = 5;
constexpr int hidden_dim = 32;
constexpr int output_dim = 1;

struct MLPWeights{
    float* d_W1; // size 32 x 3
    float* d_b1; // size 32
    float* d_W2; // size 32 x 32
    float* d_b2; // size 32
    float* d_W3; // size 1 x 32
    float* d_b3; // size 1
};

// helper for feature engineering
__device__ inline void compute_state_vector(
    const float* d_paths,
    int t,
    int path_idx,
    float prev_delta,
    const SimulationParams& params,
    float x[5]
) {
    int curr_idx = t * params.num_paths + path_idx;
    float S_t = d_paths[curr_idx];
    float tau = params.T - (t * params.dt);
    float tau_norm = tau / params.T;

    // 1-step log return
    float S_prev = (t > 0) ? d_paths[(t - 1) * params.num_paths + path_idx] : S_t;
    float r_t = logf(fmaxf(S_t, 1e-5f) / fmaxf(S_prev, 1e-5f));

    // rolling realized volatility over up to 5 steps
    int window = (t < 5) ? t : 5;
    float vol_roll = params.sigma; // fallback for inception (t = 0)

    if (window > 1) {
        float sum_r = 0.0f;
        float sum_r2 = 0.0f;
        for (int k = 0; k < window; ++k) {
            int step_curr = t - k;
            int step_past = step_curr - 1;
            float s_c = d_paths[step_curr * params.num_paths + path_idx];
            float s_p = d_paths[step_past * params.num_paths + path_idx];
            float ret = logf(fmaxf(s_c, 1e-5f) / fmaxf(s_p, 1e-5f));
            sum_r += ret;
            sum_r2 += ret * ret;
        }
        float mean_r = sum_r / static_cast<float>(window);
        float var_r = fmaxf(0.0f, (sum_r2 / static_cast<float>(window)) - (mean_r * mean_r));
        vol_roll = sqrtf(var_r / fmaxf(params.dt, 1e-6f));
    }

    x[0] = S_t / params.K;
    x[1] = prev_delta;
    x[2] = tau_norm;
    x[3] = r_t;
    x[4] = vol_roll;
}

// host interface
void forward_pass(
    float* d_paths,
    float* d_policy_deltas,
    const MLPWeights weights,
    const SimulationParams& params
);

#endif