#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include "simulation.h"
#include "mlp.h"
#include "portfolio.h"
#include "autograd.h"
#include "optimizer.h"
#include "exporter.h"

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
        
        // simulations buffer
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

        // gradient buffers
        auto d_gW1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_gb1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_gW2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_gb2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_gW3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_gb3 = cuda_utils::make_device_buffer<float>(output_dim);

        // Adam 1st moment (m) buffers
        auto d_mW1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_mb1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_mW2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_mb2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_mW3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_mb3 = cuda_utils::make_device_buffer<float>(output_dim);

        // Adam 2nd moment (v) buffers
        auto d_vW1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_vb1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_vW2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_vb2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_vW3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_vb3 = cuda_utils::make_device_buffer<float>(output_dim);

        // zero-init Adam state buffers and grad buffers
        CUDA_CHECK(cudaMemset(d_gW1.get(), 0, hidden_dim * input_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gb1.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gW2.get(), 0, hidden_dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gb2.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gW3.get(), 0, hidden_dim * output_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gb3.get(), 0, output_dim * sizeof(float)));

        CUDA_CHECK(cudaMemset(d_mW1.get(), 0, hidden_dim * input_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mb1.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mW2.get(), 0, hidden_dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mb2.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mW3.get(), 0, hidden_dim * output_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mb3.get(), 0, output_dim * sizeof(float)));

        CUDA_CHECK(cudaMemset(d_vW1.get(), 0, hidden_dim * input_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vb1.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vW2.get(), 0, hidden_dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vb2.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vW3.get(), 0, hidden_dim * output_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vb3.get(), 0, output_dim * sizeof(float)));

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

        // MLP initialization
        MLPWeights weights{
            d_W1.get(), d_b1.get(),
            d_W2.get(), d_b2.get(),
            d_W3.get(), d_b3.get()
        };

        MLPGradients grads{
            d_gW1.get(), d_gb1.get(),
            d_gW2.get(), d_gb2.get(),
            d_gW3.get(), d_gb3.get()
        };

        AdamState adam_state{
            d_mW1.get(), d_mb1.get(), d_mW2.get(), d_mb2.get(), d_mW3.get(), d_mb3.get(),
            d_vW1.get(), d_vb1.get(), d_vW2.get(), d_vb2.get(), d_vW3.get(), d_vb3.get()
        };

        AdamParams adam_params;
        adam_params.lr = 0.008f;

        std::cout << "launching generator..." << std::endl;
        generate_gbm_paths(d_paths.get(), d_payoffs.get(), d_deltas.get(), params);

        std::vector<float> h_payoffs(params.num_paths);
        CUDA_CHECK(cudaMemcpy(h_payoffs.data(), d_payoffs.get(), params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        double sum_payoffs = std::accumulate(h_payoffs.begin(), h_payoffs.end(), 0.0);
        params.option_price = std::exp(-params.r * params.T) * (sum_payoffs / params.num_paths);

        // evaluate Black-Scholes baseline
        evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_deltas.get(), d_portfolio_values.get(), params);
        std::vector<float> h_portfolio_values(params.num_paths);
        CUDA_CHECK(cudaMemcpy(h_portfolio_values.data(), d_portfolio_values.get(), params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));

        double sum_sq_error_bs = 0.0;
        for (float v : h_portfolio_values) sum_sq_error_bs += (v * v);
        double bs_mse_loss = sum_sq_error_bs / params.num_paths;

        std::cout << "black-scholes baseline MSE loss (" << params.cost_ratio * 10000 << " bps fee): " << bs_mse_loss << std::endl;

        // --- training loop
        std::cout << "mlp training..." << std::endl;
        int num_epochs = 1000;

        for(int epoch = 1; epoch <= num_epochs; ++epoch){
            // regenerate market paths each epoch
            generate_gbm_paths(d_paths.get(), d_payoffs.get(), d_deltas.get(), params, 1234ULL + epoch);

            // forward pass
            forward_pass(d_paths.get(), d_policy_deltas.get(), weights, params);

            // portfolio eval
            evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_portfolio_values.get(), params);

            // backprop through time 
            bptt_backward_pass(d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_portfolio_values.get(), weights, grads, params);

            // adam optim step
            adam_step(weights, grads, adam_state, adam_params);
            adam_params.time_step++;

            if(epoch % 10 == 0){
                CUDA_CHECK(cudaMemcpy(h_portfolio_values.data(), d_portfolio_values.get(), params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
                double sum_sq_error = 0.0;
                for (float v : h_portfolio_values) sum_sq_error += (v * v);
                double policy_mse_loss = sum_sq_error / params.num_paths;
                std::cout << "epoch [" << epoch << "/" << num_epochs << "] MSE loss: " << policy_mse_loss << std::endl;
            }
        }

        export_hedging_surface("hedging_surface.csv", weights, params);

        std::cout << "done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 