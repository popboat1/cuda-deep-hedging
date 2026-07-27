#include "exporter.h"
#include "mlp.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void evaluate_grid_kernel(
    const MLPWeights weights,
    SimulationParams params,
    float prev_delta_fixed,
    int num_s_points,
    int num_tau_points,
    float s_min, float s_max,
    float tau_min, float tau_max,
    float* d_grid_s,
    float* d_grid_tau,
    float* d_grid_policy_delta,
    float* d_grid_bs_delta
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_points = num_s_points * num_tau_points;
    if (idx >= total_points) return;

    int s_idx = idx / num_tau_points;
    int tau_idx = idx % num_tau_points;

    float s_ratio = s_min + s_idx * (s_max - s_min) / (num_s_points - 1);
    float tau_ratio = tau_min + tau_idx * (tau_max - tau_min) / (num_tau_points - 1);

    float S = s_ratio * params.K;
    float tau = tau_ratio * params.T;

    // evaluated at baseline momentum (r_t = 0.0) and baseline vol (sigma)
    float x[5] = { s_ratio, prev_delta_fixed, tau_ratio, 0.0f, params.sigma };

    // hidden layer 1
    float h1[32];
    for (int i = 0; i < hidden_dim; ++i) {
        float sum = weights.d_b1[i];
        for (int j = 0; j < input_dim; ++j) {
            sum += weights.d_W1[i * input_dim + j] * x[j];
        }
        h1[i] = fmaxf(0.0f, sum);
    }

    // hidden layer 2
    float h2[32];
    for (int i = 0; i < hidden_dim; ++i) {
        float sum = weights.d_b2[i];
        for (int j = 0; j < hidden_dim; ++j) {
            sum += weights.d_W2[i * hidden_dim + j] * h1[j];
        }
        h2[i] = fmaxf(0.0f, sum);
    }

    // output layer
    float out = weights.d_b3[0];
    for (int j = 0; j < hidden_dim; ++j) {
        out += weights.d_W3[j] * h2[j];
    }
    float policy_delta = 1.0f / (1.0f + expf(-out));

    // compute theoretical black-scholes delta
    float bs_delta = 0.0f;
    if (tau <= 1e-6f) {
        bs_delta = (S > params.K) ? 1.0f : 0.0f;
    } else {
        float sigma_sq = params.sigma * params.sigma;
        float d1 = (logf(s_ratio) + (params.r + 0.5f * sigma_sq) * tau) / (params.sigma * sqrtf(tau));
        bs_delta = normcdff(d1);
    }

    d_grid_s[idx] = s_ratio;
    d_grid_tau[idx] = tau_ratio;
    d_grid_policy_delta[idx] = policy_delta;
    d_grid_bs_delta[idx] = bs_delta;
}

void export_hedging_surface(
    const std::string& filename,
    const MLPWeights& weights,
    const SimulationParams& params,
    float prev_delta_fixed,
    int num_s_points,
    int num_tau_points
) {
    int total_points = num_s_points * num_tau_points;
    float s_min = 0.70f, s_max = 1.30f;
    float tau_min = 0.01f, tau_max = 1.00f;

    auto d_grid_s = cuda_utils::make_device_buffer<float>(total_points);
    auto d_grid_tau = cuda_utils::make_device_buffer<float>(total_points);
    auto d_grid_policy_delta = cuda_utils::make_device_buffer<float>(total_points);
    auto d_grid_bs_delta = cuda_utils::make_device_buffer<float>(total_points);

    int threads = 256;
    int blocks = cuda_utils::ceil_div(total_points, threads);

    std::cout << "sampling 2D grid surface (" << num_s_points << "x" << num_tau_points << " points) on GPU..." << std::endl;

    evaluate_grid_kernel<<<blocks, threads>>>(
        weights, params, prev_delta_fixed,
        num_s_points, num_tau_points,
        s_min, s_max, tau_min, tau_max,
        d_grid_s.get(), d_grid_tau.get(),
        d_grid_policy_delta.get(), d_grid_bs_delta.get()
    );
    CUDA_CHECK_KERNEL();

    std::vector<float> h_grid_s(total_points);
    std::vector<float> h_grid_tau(total_points);
    std::vector<float> h_grid_policy_delta(total_points);
    std::vector<float> h_grid_bs_delta(total_points);

    CUDA_CHECK(cudaMemcpy(h_grid_s.data(), d_grid_s.get(), total_points * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grid_tau.data(), d_grid_tau.get(), total_points * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grid_policy_delta.data(), d_grid_policy_delta.get(), total_points * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grid_bs_delta.data(), d_grid_bs_delta.get(), total_points * sizeof(float), cudaMemcpyDeviceToHost));

    std::ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    csv_file << "Moneyness,Tau_Norm,Policy_Delta,BS_Delta\n";
    for (int i = 0; i < total_points; ++i) {
        csv_file << h_grid_s[i] << ","
                 << h_grid_tau[i] << ","
                 << h_grid_policy_delta[i] << ","
                 << h_grid_bs_delta[i] << "\n";
    }

    csv_file.close();
    std::cout << "successfully exported surface data to: " << filename << std::endl;
}