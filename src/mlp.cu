#include "mlp.cuh"
#include <cmath>

__constant__ float c_W1[hidden_dim * input_dim];
__constant__ float c_b1[hidden_dim];
__constant__ float c_W2[hidden_dim * hidden_dim];
__constant__ float c_b2[hidden_dim];
__constant__ float c_W3[hidden_dim * output_dim];
__constant__ float c_b3[output_dim];

void copy_weights_to_constant(const MLPWeights& weights) {
    // copy mlp weights to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_W1, weights.d_W1, hidden_dim * input_dim * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_b1, weights.d_b1, hidden_dim * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_W2, weights.d_W2, hidden_dim * hidden_dim * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_b2, weights.d_b2, hidden_dim * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_W3, weights.d_W3, hidden_dim * output_dim * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_b3, weights.d_b3, output_dim * sizeof(float), 0, cudaMemcpyDeviceToDevice));
}

__global__ void forward(
        float* d_paths, 
        float* d_policy_deltas, 
        SimulationParams params
){
    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_idx >= params.num_paths) return;

    // fetch prev hedge position
    float prev_delta = 0.0f;

    // rebalance at t = 0 .. num_steps - 2; expiry t = num_steps - 1 liquidates
    for (int t = 0; t < params.num_steps - 1; ++t) {
        int curr_idx = t * params.num_paths + path_idx;

        // 5 normalized state features
        float x[5];
        compute_state_vector(d_paths, t, path_idx, prev_delta, params, x);

        // hidden layer 1 (input_dim -> hidden_dim + ReLU)
        float h1[hidden_dim];
        for (int i = 0; i < hidden_dim; ++i) {
            float sum = c_b1[i];
            for (int j = 0; j < input_dim; ++j) {
                sum += c_W1[i * input_dim + j] * x[j];
            }
            h1[i] = fmaxf(0.0f, sum); // ReLU
        }

        // hidden layer 2 (hidden_dim -> hidden_dim + ReLU)
        float h2[hidden_dim];
        for (int i = 0; i < hidden_dim; ++i) {
            float sum = c_b2[i];
            for (int j = 0; j < hidden_dim; ++j) {
                sum += c_W2[i * hidden_dim + j] * h1[j];
            }
            h2[i] = fmaxf(0.0f, sum); // ReLU
        }

        // output layer (hidden_dim -> 1 + sigmoid)
        float out = c_b3[0];
        for (int j = 0; j < hidden_dim; ++j) {
            out += c_W3[j] * h2[j];
        }

        float policy_delta = 1.0f / (1.0f + expf(-out));
        d_policy_deltas[curr_idx] = policy_delta;
        prev_delta = policy_delta;
    }

    // zero target delta at expiry (tau = 0)
    if (params.num_steps > 0) {
        d_policy_deltas[(params.num_steps - 1) * params.num_paths + path_idx] = 0.0f;
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

    copy_weights_to_constant(weights);

    forward<<<blocks, threads>>>(
        d_paths,
        d_policy_deltas,
        params
    );
    CUDA_CHECK_KERNEL();
}
