#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include "simulation.h"
#include "mlp.h"
#include "portfolio.h"

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
        params.cost_ratio = 0.001f;

        // query seed
        std::mt19937 rng(42);

        std::cout << "allocating vram for " << params.num_paths << " paths..." << std::endl;
        
        // allocate buffer
        size_t total_elements = params.num_paths * params.num_steps;
        auto d_paths = cuda_utils::make_device_buffer<float>(total_elements);
        auto d_payoffs = cuda_utils::make_device_buffer<float>(params.num_paths);
        auto d_deltas = cuda_utils::make_device_buffer<float>(params.num_paths * params.num_steps);
        auto d_policy_deltas = cuda_utils::make_device_buffer<float>(params.num_paths * params.num_steps);
        auto d_portfolio_values = cuda_utils::make_device_buffer<float>(params.num_paths);

        // buffers for weights and biases
        auto d_W1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_b1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_W2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_b2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_W3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_b3 = cuda_utils::make_device_buffer<float>(output_dim);

        // host vectors to store weights and biases
        std::vector<float> h_W1(hidden_dim * input_dim);
        std::vector<float> h_b1(hidden_dim);
        std::vector<float> h_W2(hidden_dim * hidden_dim);
        std::vector<float> h_b2(hidden_dim);
        std::vector<float> h_W3(hidden_dim * output_dim);
        std::vector<float> h_b3(output_dim);

        // init weights with Xavier init
        auto fill_xavier = [&](std::vector<float>& vec, float std_dev) {
            std::normal_distribution<float> dist(0.0f, std_dev);
            for (auto& val : vec) val = dist(rng);
        };

        fill_xavier(h_W1, std::sqrt(2.0f / (input_dim + hidden_dim)));
        fill_xavier(h_W2, std::sqrt(2.0f / (hidden_dim + hidden_dim)));
        fill_xavier(h_W3, std::sqrt(2.0f / (hidden_dim + output_dim)));

        // --- copy weights to device
        CUDA_CHECK(cudaMemcpy(d_W1.get(), h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1.get(), h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2.get(), h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2.get(), h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W3.get(), h_W3.data(), h_W3.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3.get(), h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

        MLPWeights weights{
            d_W1.get(), d_b1.get(),
            d_W2.get(), d_b2.get(),
            d_W3.get(), d_b3.get()
        };

        std::cout << "launching generator..." << std::endl;
        generate_gbm_paths(d_paths.get(), d_payoffs.get(), d_deltas.get(), params);

        std::cout << "running neural policy network forward pass..." << std::endl;
        forward_pass(d_paths.get(), d_policy_deltas.get(), weights, params);    

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

        std::vector<float> h_deltas(params.num_paths * params.num_steps);
        CUDA_CHECK(cudaMemcpy(
            h_deltas.data(),
            d_deltas.get(),
            params.num_paths * params.num_steps * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        std::vector<float> h_policy_deltas(params.num_paths * params.num_steps);
        CUDA_CHECK(cudaMemcpy(
            h_policy_deltas.data(),
            d_policy_deltas.get(),
            params.num_paths * params.num_steps * sizeof(float),
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
        params.option_price = mc_call_price;

        double simulated_mean_ST = sum_ST / params.num_paths;
        double theoretical_mean_ST = params.S0 * std::exp(params.r * params.T);

        std::cout << "simulated  mean S_T: " << simulated_mean_ST << std::endl;
        std::cout << "theoretical mean S_T: " << theoretical_mean_ST << std::endl;

        // validate delta at t = 0
        std::cout << "initial black-scholes delta (t=0): " << h_deltas[0] << std::endl;

        // validate terminal delta at t = N-1 for path 0
        int path0_terminal_idx = (0 * params.num_steps) + (params.num_steps - 1);
        std::cout << "path 0 terminal price: " << h_paths[path0_terminal_idx] << std::endl;
        std::cout << "path 0 terminal delta: " << h_deltas[path0_terminal_idx] << std::endl;

        // validate policy deltas
        std::cout << "initial policy delta (t=0): " << h_policy_deltas[0] << std::endl;
        std::cout << "path 0 terminal policy delta (t=N-1): " << h_policy_deltas[path0_terminal_idx] << std::endl;

        // validate portfolio
        evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_deltas.get(), d_portfolio_values.get(), params);

        std::vector<float> h_portfolio_values(params.num_paths);
        CUDA_CHECK(cudaMemcpy(
            h_portfolio_values.data(),
            d_portfolio_values.get(),
            params.num_paths * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        // ---------- calculate Black-Scholes MSE Loss
        double sum_sq_error_bs = 0.0;
        for (int i = 0; i < params.num_paths; ++i) {
            float V_T = h_portfolio_values[i];
            sum_sq_error_bs += (V_T * V_T);
        }
        double mse_loss_bs = sum_sq_error_bs / params.num_paths;

        std::cout << "black-scholes hedging MSE loss (" << params.cost_ratio * 10000 << "bps fee): " << mse_loss_bs << std::endl;

        // evaluate untrained neural policy network
        evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_portfolio_values.get(), params);

        CUDA_CHECK(cudaMemcpy(
            h_portfolio_values.data(),
            d_portfolio_values.get(),
            params.num_paths * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        double sum_sq_error_policy = 0.0;
        for (int i = 0; i < params.num_paths; ++i) {
            float V_T = h_portfolio_values[i];
            sum_sq_error_policy += (V_T * V_T);
        }
        double mse_loss_policy = sum_sq_error_policy / params.num_paths;

        std::cout << "untrained policy hedging MSE loss: " << mse_loss_policy << std::endl;

        std::cout << "done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 