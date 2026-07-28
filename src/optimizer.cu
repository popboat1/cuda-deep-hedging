#include "optimizer.h"
#include <cmath>

// store MLP pointers & hyperparams in constant memory
__constant__ MLPWeights c_weights;
__constant__ AdamParams c_adam_params;

// static device weight pointer initialization definition
void init_optimizer_weights(const MLPWeights& weights) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights, &weights, sizeof(MLPWeights)));
}

// generic element-wise Adam kernel
__global__ void adam_update(
    MLPGradients grads,
    AdamState state,
    float bias_corr1,
    float bias_corr2
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    constexpr int sz_W1 = hidden_dim * input_dim;   // 160
    constexpr int sz_b1 = hidden_dim;              // 32
    constexpr int sz_W2 = hidden_dim * hidden_dim; // 1024
    constexpr int sz_b2 = hidden_dim;              // 32
    constexpr int sz_W3 = hidden_dim * output_dim; // 32
    constexpr int sz_b3 = output_dim;              // 1
    constexpr int total_params = sz_W1 + sz_b1 + sz_W2 + sz_b2 + sz_W3 + sz_b3; // 1281

    if (idx >= total_params) return;

    float* param = nullptr;
    float* grad  = nullptr;
    float* m     = nullptr;
    float* v     = nullptr;

    // map thread index to parameter pointers stored in __constant__ memory
    if (idx < sz_W1) {
        int offset = idx;
        param = c_weights.d_W1 + offset;
        grad  = grads.d_gW1 + offset;
        m     = state.d_mW1 + offset;
        v     = state.d_vW1 + offset;
    } else if (idx < sz_W1 + sz_b1) {
        int offset = idx - sz_W1;
        param = c_weights.d_b1 + offset;
        grad  = grads.d_gb1 + offset;
        m     = state.d_mb1 + offset;
        v     = state.d_vb1 + offset;
    } else if (idx < sz_W1 + sz_b1 + sz_W2) {
        int offset = idx - (sz_W1 + sz_b1);
        param = c_weights.d_W2 + offset;
        grad  = grads.d_gW2 + offset;
        m     = state.d_mW2 + offset;
        v     = state.d_vW2 + offset;
    } else if (idx < sz_W1 + sz_b1 + sz_W2 + sz_b2) {
        int offset = idx - (sz_W1 + sz_b1 + sz_W2);
        param = c_weights.d_b2 + offset;
        grad  = grads.d_gb2 + offset;
        m     = state.d_mb2 + offset;
        v     = state.d_vb2 + offset;
    } else if (idx < sz_W1 + sz_b1 + sz_W2 + sz_b2 + sz_W3) {
        int offset = idx - (sz_W1 + sz_b1 + sz_W2 + sz_b2);
        param = c_weights.d_W3 + offset;
        grad  = grads.d_gW3 + offset;
        m     = state.d_mW3 + offset;
        v     = state.d_vW3 + offset;
    } else {
        int offset = idx - (sz_W1 + sz_b1 + sz_W2 + sz_b2 + sz_W3);
        param = c_weights.d_b3 + offset;
        grad  = grads.d_gb3 + offset;
        m     = state.d_mb3 + offset;
        v     = state.d_vb3 + offset;
    }

    float g = *grad;
    float m_val = *m;
    float v_val = *v;

    // Adam moment updates
    m_val = c_adam_params.beta1 * m_val + (1.0f - c_adam_params.beta1) * g;
    v_val = c_adam_params.beta2 * v_val + (1.0f - c_adam_params.beta2) * (g * g);

    *m = m_val;
    *v = v_val;

    float m_hat = m_val / bias_corr1;
    float v_hat = v_val / bias_corr2;
    
    // update model weight in global memory
    *param -= c_adam_params.lr * m_hat / (sqrtf(v_hat) + c_adam_params.epsilon);

    // zero grad
    *grad = 0.0f;
}

void adam_step(
    MLPWeights& weights,
    MLPGradients& grads,
    AdamState& state,
    AdamParams& params
) {
    // copy weight descriptors and Adam prams into constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_adam_params, &params, sizeof(AdamParams)));

    // pre-calculate bias correction denominators
    float bias_corr1 = 1.0f - std::pow(params.beta1, static_cast<float>(params.time_step));
    float bias_corr2 = 1.0f - std::pow(params.beta2, static_cast<float>(params.time_step));
    constexpr int total_params = (hidden_dim * input_dim) + hidden_dim +
                                 (hidden_dim * hidden_dim) + hidden_dim +
                                 (hidden_dim * output_dim) + output_dim;
    
    int threads = 256;
    int blocks = cuda_utils::ceil_div(total_params, threads);

    // launch kernel
    adam_update<<<blocks, threads>>>(grads, state, bias_corr1, bias_corr2);
    CUDA_CHECK_KERNEL();
}