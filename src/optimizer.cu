#include "optimizer.h"
#include <cmath>

// generic element-wise Adam kernel
__global__ void adam_update(
    float* param,
    float* grad,
    float* m,
    float* v,
    int size,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float bias_corr1,
    float bias_corr2
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    float g = grad[idx];
    float m_val = m[idx];
    float v_val = v[idx];

    // update 1st and 2nd moments
    m_val = beta1 * m_val + (1.0f - beta1) * g;
    v_val = beta2 * v_val + (1.0f - beta2) * (g * g);

    // save moments
    m[idx] = m_val;
    v[idx] = v_val;

    // compute bias-corrected moment estimates
    float m_hat = m_val / bias_corr1;
    float v_hat = v_val / bias_corr2;

    // update param in-place
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);

    // zero grad
    grad[idx] = 0.0f;
}

// launcher helper
static void update_tensor(
    float* d_param,
    float* d_grad,
    float* d_m,
    float* d_v,
    int size,
    const AdamParams& params,
    float bias_corr1,
    float bias_corr2
) {
    int threads = 256;
    int blocks = cuda_utils::ceil_div(size, threads);

    adam_update<<<blocks, threads>>>(
        d_param, d_grad, d_m, d_v, size,
        params.lr, params.beta1, params.beta2, params.epsilon,
        bias_corr1, bias_corr2
    );
    CUDA_CHECK_KERNEL();
}

void adam_step(
    MLPWeights& weights,
    MLPGradients& grads,
    AdamState& state,
    AdamParams& params
) {
    // pre-calculate bias correction denominators
    float bias_corr1 = 1.0f - std::pow(params.beta1, static_cast<float>(params.time_step));
    float bias_corr2 = 1.0f - std::pow(params.beta2, static_cast<float>(params.time_step));

    // layer 1
    update_tensor(weights.d_W1, grads.d_gW1, state.d_mW1, state.d_vW1, hidden_dim * input_dim, params, bias_corr1, bias_corr2);
    update_tensor(weights.d_b1, grads.d_gb1, state.d_mb1, state.d_vb1, hidden_dim, params, bias_corr1, bias_corr2);

    // layer 2
    update_tensor(weights.d_W2, grads.d_gW2, state.d_mW2, state.d_vW2, hidden_dim * hidden_dim, params, bias_corr1, bias_corr2);
    update_tensor(weights.d_b2, grads.d_gb2, state.d_mb2, state.d_vb2, hidden_dim, params, bias_corr1, bias_corr2);

    // layer 3
    update_tensor(weights.d_W3, grads.d_gW3, state.d_mW3, state.d_vW3, output_dim * hidden_dim, params, bias_corr1, bias_corr2);
    update_tensor(weights.d_b3, grads.d_gb3, state.d_mb3, state.d_vb3, output_dim, params, bias_corr1, bias_corr2);
}