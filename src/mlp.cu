#include "mlp.h"
#include <cmath>

__global__ void forward(
        float* d_paths, 
        float* d_policy_deltas, 
        MLPWeights weights, 
        SimulationParams params
){
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_idx >= params.num_paths) return;

    int path_offset = path_idx * params.num_steps;
    // fetch prev hedge position
    float prev_delta = 0.0f;

    for(int t = 0; t < params.num_steps; ++t){
        // get curr stock price
        int curr_idx = t * params.num_paths + path_idx;
        float S_t = d_paths[curr_idx];
    
        // compute tau (time-to-expiry)
        float tau = params.T - (t * params.dt);
        float tau_norm = tau / params.T;
    
        // state in vector
        float x[3] = { S_t / params.K, prev_delta, tau_norm };
    
        // hidden layer 1 (3 -> 32 + ReLU)
        float h1[32];
        for (int i = 0; i < hidden_dim; ++i){
            float sum = weights.d_b1[i];
            for (int j = 0; j < input_dim; ++j){
                sum += weights.d_W1[i * input_dim + j] * x[j];
            }
            h1[i] = fmaxf(0.0f, sum); // ReLU
        }
    
        // hidden layer 2 (32 -> 32 + ReLU)
        float h2[32];
        for (int i = 0; i < hidden_dim; ++i){
            float sum = weights.d_b2[i];
            for(int j = 0; j < hidden_dim; ++j){
                sum += weights.d_W2[i * hidden_dim + j] * h1[j];
            }
            h2[i] = fmaxf(0.0f, sum); // ReLU
        }
    
        // output layer (32 -> 1 + sigmoid)
        float out = weights.d_b3[0];
        for(int j = 0; j < hidden_dim; ++j){
            out += weights.d_W3[j] * h2[j];
        }
    
        // sigmoid
        float policy_delta = 1.0f / (1.0f + expf(-out));
        // store output delta & update prev_delta for next step
        d_policy_deltas[curr_idx] = policy_delta;
        prev_delta = policy_delta;
    }
}

void forward_pass(
    float* d_paths,
    float* d_policy_deltas,
    const MLPWeights weights,
    const SimulationParams& params
){
    int threads = 256;
    int blocks = cuda_utils::ceil_div(params.num_paths, threads);

    forward<<<blocks, threads>>>(
        d_paths,
        d_policy_deltas,
        weights,
        params
    );
    CUDA_CHECK_KERNEL();
}