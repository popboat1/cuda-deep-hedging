#ifndef MLP_CUH
#define MLP_CUH

#include "simulation.cuh"

constexpr int input_dim = 5;
constexpr int hidden_dim = 64;
constexpr int output_dim = 1;
constexpr float d_x1_d_prev_delta = 2.0f;

struct MLPWeights{
    float* d_W1; // size hidden_dim x input_dim
    float* d_b1; // size hidden_dim
    float* d_W2; // size hidden_dim x hidden_dim
    float* d_b2; // size hidden_dim
    float* d_W3; // size output_dim x hidden_dim
    float* d_b3; // size output_dim
};

extern __constant__ float c_W1[hidden_dim * input_dim];
extern __constant__ float c_b1[hidden_dim];
extern __constant__ float c_W2[hidden_dim * hidden_dim];
extern __constant__ float c_b2[hidden_dim];
extern __constant__ float c_W3[hidden_dim * output_dim];
extern __constant__ float c_b3[output_dim];

// copy weights from global memory to constant memory
void copy_weights_to_constant(const MLPWeights& weights);

// shared zero-centered O(1) feature normalization for trajectory & grid eval
__device__ inline void normalize_state_features(
    float S_t,
    float prev_delta,
    float tau,
    float r_t,
    float vol_roll,
    const SimulationParams& params,
    float x[5]
) {
    float total_vol = fmaxf(params.sigma * sqrtf(fmaxf(params.T, 1e-8f)), 1e-4f);
    float step_vol = fmaxf(params.sigma * sqrtf(fmaxf(params.dt, 1e-8f)), 1e-6f);
    float tau_norm = tau / fmaxf(params.T, 1e-8f);

    x[0] = logf(fmaxf(S_t, 1e-5f) / fmaxf(params.K, 1e-5f)) / total_vol;
    x[1] = d_x1_d_prev_delta * prev_delta - 1.0f;
    x[2] = 2.0f * tau_norm - 1.0f;
    x[3] = r_t / step_vol;
    x[4] = (vol_roll / fmaxf(params.sigma, 1e-6f)) - 1.0f;
}

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

    // 1-step log return
    float S_prev = (t > 0) ? d_paths[(t - 1) * params.num_paths + path_idx] : S_t;
    float r_t = logf(fmaxf(S_t, 1e-5f) / fmaxf(S_prev, 1e-5f));

    // rolling realized volatility over up to 5 steps with prior blend at t = 0, 1
    constexpr int vol_window_max = 5;
    int window = (t < vol_window_max) ? t : vol_window_max;
    float prior_var = params.sigma * params.sigma * params.dt;
    float sum_r2 = 0.0f;

    for (int k = 0; k < window; ++k) {
        int step_curr = t - k;
        int step_past = step_curr - 1;
        float s_c = d_paths[step_curr * params.num_paths + path_idx];
        float s_p = d_paths[step_past * params.num_paths + path_idx];
        float ret = logf(fmaxf(s_c, 1e-5f) / fmaxf(s_p, 1e-5f));
        sum_r2 += ret * ret;
    }

    float var_r = (static_cast<float>(vol_window_max - window) * prior_var + sum_r2)
                  / static_cast<float>(vol_window_max);
    float vol_roll = sqrtf(fmaxf(0.0f, var_r) / fmaxf(params.dt, 1e-8f));

    normalize_state_features(S_t, prev_delta, tau, r_t, vol_roll, params, x);
}

// host interface
void forward_pass(
    float* d_paths,
    float* d_policy_deltas,
    const MLPWeights weights,
    const SimulationParams& params
);

#endif
