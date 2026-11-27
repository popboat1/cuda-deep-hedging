#ifndef MLP_CUH
#define MLP_CUH

#include "simulation.cuh"

constexpr int input_dim = 5;
constexpr int hidden_dim = 32;
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

// shared zero-centered O(1) feature normalization for trajectory and grid eval
__device__ inline void normalize_state_features(
    float S_t,
    float prev_delta,
    float tau,
    float r_t,
    float inst_vol,
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
    x[4] = (inst_vol / fmaxf(params.sigma, 1e-6f)) - 1.0f;
}

// build state vector using simulated instantaneous volatility directly
__device__ inline void compute_state_vector(
    const float* d_paths,
    const float* d_vols,
    int t,
    int path_idx,
    float prev_delta,
    const SimulationParams& params,
    float x[5]
) {
    int curr_idx = t * params.num_paths + path_idx;
    float S_t = d_paths[curr_idx];
    float inst_vol = d_vols[curr_idx];
    float tau = params.T - (t * params.dt);

    float S_prev = (t > 0) ? d_paths[(t - 1) * params.num_paths + path_idx] : S_t;
    float r_t = logf(fmaxf(S_t, 1e-5f) / fmaxf(S_prev, 1e-5f));

    normalize_state_features(S_t, prev_delta, tau, r_t, inst_vol, params, x);
}

// host interface
void forward_pass(
    const float* d_paths,
    const float* d_vols,
    float* d_policy_deltas,
    const MLPWeights weights,
    const SimulationParams& params
);

#endif
