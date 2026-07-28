#include "autograd.h"
#include "mlp.h"
#include <cmath>

__device__ inline float signf(float x) {
    if (x > 0.0f) return 1.0f;
    if (x < 0.0f) return -1.0f;
    return 0.0f;
}

// warp reduction using shuffle instructions
__device__ inline float warpReduceSum(float val) {
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// 
__device__ inline void warp_atomic_add(float* address, float val) {
    unsigned int mask = __activemask();
    val = warpReduceSum(val);
    int leader = __ffs(mask) - 1;
    if ((threadIdx.x & 31) == leader) {
        atomicAdd(address, val);
    }
}

__global__ void bptt_backward(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_policy_deltas,
    const float* d_portfolio_values,
    const MLPWeights weights,
    MLPGradients grads,
    const SimulationParams params,
    float var_cutoff,
    float cvar_alpha,
    float lambda_cvar,
    bool use_hybrid
){
    // --- shared memory buffers for block grad accum
    __shared__ float s_gW1[hidden_dim * input_dim];
    __shared__ float s_gb1[hidden_dim];
    __shared__ float s_gW2[hidden_dim * hidden_dim];
    __shared__ float s_gb2[hidden_dim];
    __shared__ float s_gW3[hidden_dim * output_dim];
    __shared__ float s_gb3[output_dim];

    int tid = threadIdx.x;

    // zero-out shared memory across the block
    for (int i = tid; i < hidden_dim * input_dim; i += blockDim.x) s_gW1[i] = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) s_gb1[i] = 0.0f;
    for (int i = tid; i < hidden_dim * hidden_dim; i += blockDim.x) s_gW2[i] = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) s_gb2[i] = 0.0f;
    for (int i = tid; i < hidden_dim * output_dim; i += blockDim.x) s_gW3[i] = 0.0f;
    for (int i = tid; i < output_dim; i += blockDim.x) s_gb3[i] = 0.0f;

    __syncthreads();

    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    bool active = (path_idx < params.num_paths);

    if (active) {
        float V_T = d_portfolio_values[path_idx];
        
        // base MSE gradient active across 100% of paths
        float dL_dV_T = (2.0f / params.num_paths) * V_T;

        // add CVaR tail penalty gradient if path falls in worst 5% tail
        if (use_hybrid) {
            float tail_prob = 1.0f - cvar_alpha; // e.g. 0.05
            if (V_T <= var_cutoff) {
                dL_dV_T += lambda_cvar * (-1.0f / (params.num_paths * tail_prob));
            }
        }

        float exp_r_dt = expf(params.r * params.dt);
        float upstream_d_delta = 0.0f;

        for (int t = params.num_steps - 1; t >= 0; --t) {
            int curr_idx = t * params.num_paths + path_idx;
            float S_t = d_paths[curr_idx];
            float delta_t = d_policy_deltas[curr_idx];

            // --- fetch prev position & re-evaluate activations
            float prev_delta = (t > 0) ? d_policy_deltas[(t - 1) * params.num_paths + path_idx] : 0.0f;
            
            // recompute 5-feature input state vector
            float x[5];
            compute_state_vector(d_paths, t, path_idx, prev_delta, params, x);

            // recompute layer 1
            float h1[32];
            for (int i = 0; i < hidden_dim; ++i) {
                float sum = weights.d_b1[i];
                for (int j = 0; j < input_dim; ++j) {
                    sum += weights.d_W1[i * input_dim + j] * x[j];
                }
                h1[i] = fmaxf(0.0f, sum);
            }

            // recompute layer 2
            float h2[32];
            for (int i = 0; i < hidden_dim; ++i) {
                float sum = weights.d_b2[i];
                for (int j = 0; j < hidden_dim; ++j) {
                    sum += weights.d_W2[i * hidden_dim + j] * h1[j];
                }
                h2[i] = fmaxf(0.0f, sum);
            }

            // --- compute final portfolio derivative
            int steps_from_next_to_expiry = (params.num_steps - 1) - t;
            float accrual_to_T = expf(params.r * steps_from_next_to_expiry * params.dt);

            float sign_t = signf(delta_t - prev_delta);
            float dVT_ddelta = 0.0f;

            if (t == params.num_steps - 1) {
                // at terminal step, stock is liquidated into payoff
                dVT_ddelta = S_t * (1.0f - (1.0f + params.cost_ratio * sign_t) * exp_r_dt);
            } else {
                // intermediate step: cash drain accrued to time T
                int next_idx = (t + 1) * params.num_paths + path_idx;
                float S_next = d_paths[next_idx];
                float next_delta = d_policy_deltas[next_idx];
                float sign_next = signf(next_delta - delta_t);

                float step_return = -S_t * (1.0f + params.cost_ratio * sign_t) * exp_r_dt
                                    + S_next * (1.0f + params.cost_ratio * sign_next);

                dVT_ddelta = step_return * accrual_to_T;
            }

            // combined total gradient for delta_t
            float dL_ddelta = (dL_dV_T * dVT_ddelta) + upstream_d_delta;

            // --- backpropagation
            // out layer
            float d3 = dL_ddelta * (delta_t * (1.0f - delta_t));

            // hidden layer 2
            float d2[32];
            for (int i = 0; i < hidden_dim; ++i) {
                float incoming = weights.d_W3[i] * d3;
                d2[i] = (h2[i] > 0.0f) ? incoming : 0.0f;
            }

            // hidden layer 1
            float d1[32];
            for (int i = 0; i < hidden_dim; ++i) {
                float incoming = 0.0f;
                for (int j = 0; j < hidden_dim; ++j) {
                    incoming += weights.d_W2[j * hidden_dim + i] * d2[j]; // transpose: W2[j, i]
                }
                d1[i] = (h1[i] > 0.0f) ? incoming : 0.0f;
            }

            // compute upstream gradient for prev_delta to pass to step t-1
            upstream_d_delta = 0.0f;
            for (int i = 0; i < hidden_dim; ++i) {
                upstream_d_delta += weights.d_W1[i * input_dim + 1] * d1[i];
            }

            // --- gradient accumulation into shared memory
            // layer 3 grad
            warp_atomic_add(&s_gb3[0], d3);
            for (int j = 0; j < hidden_dim; ++j) {
                warp_atomic_add(&s_gW3[j], d3 * h2[j]);
            }

            // layer 2 grad
            for (int i = 0; i < hidden_dim; ++i) {
                warp_atomic_add(&s_gb2[i], d2[i]);
                for (int j = 0; j < hidden_dim; ++j) {
                    warp_atomic_add(&s_gW2[i * hidden_dim + j], d2[i] * h1[j]);
                }
            }

            // layer 1 grad
            for (int i = 0; i < hidden_dim; ++i) {
                warp_atomic_add(&s_gb1[i], d1[i]);
                for (int j = 0; j < input_dim; ++j) {
                    warp_atomic_add(&s_gW1[i * input_dim + j], d1[i] * x[j]);
                }
            }
        }
    }

    // sync block before writing to global memory
    __syncthreads();

    // flush block shared memory gradients to global memory
    for (int i = tid; i < hidden_dim * input_dim; i += blockDim.x) atomicAdd(&grads.d_gW1[i], s_gW1[i]);
    for (int i = tid; i < hidden_dim; i += blockDim.x) atomicAdd(&grads.d_gb1[i], s_gb1[i]);
    for (int i = tid; i < hidden_dim * hidden_dim; i += blockDim.x) atomicAdd(&grads.d_gW2[i], s_gW2[i]);
    for (int i = tid; i < hidden_dim; i += blockDim.x) atomicAdd(&grads.d_gb2[i], s_gb2[i]);
    for (int i = tid; i < hidden_dim * output_dim; i += blockDim.x) atomicAdd(&grads.d_gW3[i], s_gW3[i]);
    for (int i = tid; i < output_dim; i += blockDim.x) atomicAdd(&grads.d_gb3[i], s_gb3[i]);
}

void bptt_backward_pass(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_policy_deltas,
    const float* d_portfolio_values,
    const MLPWeights& weights,
    MLPGradients& grads,
    const SimulationParams& params,
    float var_cutoff,
    float cvar_alpha,
    float lambda_cvar,
    bool use_hybrid
){
    int threads = 256;
    int blocks = cuda_utils::ceil_div(params.num_paths, threads);

    bptt_backward<<<blocks, threads>>>(
        d_paths, d_payoffs, d_policy_deltas, d_portfolio_values, 
        weights, grads, params, var_cutoff, cvar_alpha, lambda_cvar, use_hybrid
    );
}