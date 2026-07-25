#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include "simulation.h"

int main() {
    try {
        SimulationParams params;
        params.num_paths = 10000;
        params.num_steps = 30;
        params.S0 = 100.0f;
        params.r = 0.05f;
        params.sigma = 0.2f;
        params.T = 1.0f / 12.0f; // 1 month
        params.dt = params.T / params.num_steps;
        params.K = 100.0f; // strike price (at the money)

        std::cout << "allocating vram for " << params.num_paths << " paths..." << std::endl;
        
        // allocate buffer
        size_t total_elements = params.num_paths * params.num_steps;
        auto d_paths = cuda_utils::make_device_buffer<float>(total_elements);
        auto d_payoffs = cuda_utils::make_device_buffer<float>(params.num_paths);

        std::cout << "launching generator..." << std::endl;
        generate_gbm_paths(d_paths.get(), d_payoffs.get(), params);

        std::cout << "validating on cpu..." << std::endl;

        std::vector<float> h_paths(total_elements);
        CUDA_CHECK(cudaMemcpy(
            h_paths.data(),
            d_paths.get(),
            total_elements * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        std::vector<float> h_payoffs(params.num_paths);
        CUDA_CHECK(cudaMemcpy(
            h_payoffs.data(),
            d_payoffs.get(),
            params.num_paths * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        double sum_ST = 0.0;
        for (int i = 0; i < params.num_paths; ++i) {
            int final_step_idx = (i * params.num_steps) + (params.num_steps - 1);
            sum_ST += h_paths[final_step_idx];
        }

        // ---------- option price calculations
        double sum_payoffs = std::accumulate(h_payoffs.begin(), h_payoffs.end(), 0.0);
        double mean_payoff = sum_payoffs / params.num_paths;

        double mc_call_price = std::exp(-params.r * params.T) * mean_payoff;

        std::cout << "Monte Carlo Call Price: " << mc_call_price << std::endl;

        double simulated_mean_ST = sum_ST / params.num_paths;
        double theoretical_mean_ST = params.S0 * std::exp(params.r * params.T);

        std::cout << "simulated  mean S_T: " << simulated_mean_ST << std::endl;
        std::cout << "theoretical mean S_T: " << theoretical_mean_ST << std::endl;

        std::cout << "done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 