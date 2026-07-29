#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <filesystem>
#include <algorithm>

#include "simulation.cuh"
#include "mlp.cuh"
#include "portfolio.cuh"
#include "autograd.cuh"
#include "optimizer.cuh"
#include "exporter.cuh"
#include "dataloader.h"

int main() {
    try {
        // guard create results directory if missing
        std::filesystem::create_directories("../results");
        
        SimulationParams params;
        params.num_steps = 30;
        params.S0 = 100.0f;
        params.r = 0.04f;
        params.sigma = 0.18f; 
        params.T = 30.0f / (365.0f * 24.0f); // (30 / 8760 years)
        params.dt = params.T / (params.num_steps - 1);
        params.K = 100.0f; // strike price (at the money)
        params.cost_ratio = 0.0010f;

        // query seed
        std::mt19937 rng(42);

        RealDataset train_ds = load_and_normalize_real_csv("../data/GSPC_train_paths.csv", params);
        params.num_paths = train_ds.num_paths;

        std::cout << "allocating vram for " << params.num_paths << " paths..." << std::endl;
        
        // simulations buffer
        size_t total_elements = params.num_paths * params.num_steps;
        auto d_paths = cuda_utils::make_device_buffer<float>(total_elements);
        auto d_payoffs = cuda_utils::make_device_buffer<float>(params.num_paths);
        auto d_deltas = cuda_utils::make_device_buffer<float>(params.num_paths * params.num_steps);
        auto d_policy_deltas = cuda_utils::make_device_buffer<float>(params.num_paths * params.num_steps);
        auto d_portfolio_values = cuda_utils::make_device_buffer<float>(params.num_paths);

        // copy real data to VRAM
        CUDA_CHECK(cudaMemcpy(d_paths.get(), train_ds.h_paths.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_payoffs.get(), train_ds.h_payoffs.data(), params.num_paths * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_deltas.get(), train_ds.h_deltas.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));

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

        // host checkpoint vectors for best weights
        std::vector<float> best_W1(hidden_dim * input_dim);
        std::vector<float> best_b1(hidden_dim);
        std::vector<float> best_W2(hidden_dim * hidden_dim);
        std::vector<float> best_b2(hidden_dim);
        std::vector<float> best_W3(hidden_dim * output_dim);
        std::vector<float> best_b3(output_dim);

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
        float initial_lr = 0.005f;
        float min_lr     = 0.0001f;

        std::vector<float> h_payoffs(params.num_paths);
        CUDA_CHECK(cudaMemcpy(h_payoffs.data(), d_payoffs.get(), params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));

        // compute option price directly from training dataset
        double sum_payoffs = std::accumulate(train_ds.h_payoffs.begin(), train_ds.h_payoffs.end(), 0.0);
        params.option_price = std::exp(-params.r * params.T) * (sum_payoffs / params.num_paths);

        // evaluate Black-Scholes baseline
        evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_deltas.get(), d_portfolio_values.get(), params);
        std::vector<float> h_portfolio_values(params.num_paths);
        CUDA_CHECK(cudaMemcpy(h_portfolio_values.data(), d_portfolio_values.get(), params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));

        double sum_sq_error_bs = 0.0;
        for (float v : h_portfolio_values) sum_sq_error_bs += (v * v);
        double bs_mse_loss = sum_sq_error_bs / params.num_paths;

        std::cout << "black-scholes baseline MSE loss (" << params.cost_ratio * 10000 << " bps fee): " << bs_mse_loss << std::endl;

        // --- training loop parameters for Hybrid Loss (MSE + Lambda * CVaR)
        float cvar_alpha = 0.98f;    //  CVaR tail level
        float lambda_cvar = 1.25f;   // CVaR penalty multiplier
        bool use_hybrid = true;

        init_optimizer_weights(weights);

        std::cout << "mlp training with Hybrid Loss (MSE + " << lambda_cvar << " * "<< cvar_alpha * 100 << "% CVaR)..." << std::endl;
        int num_epochs = 2000;
        double best_loss = std::numeric_limits<double>::max();
        int best_epoch = 0;

        for(int epoch = 1; epoch <= num_epochs; ++epoch){
            // cosine annealing lr schedule
            adam_params.lr = min_lr + 0.5f * (initial_lr - min_lr) * 
                     (1.0f + std::cos(M_PI * epoch / num_epochs));

            // forward pass
            forward_pass(d_paths.get(), d_policy_deltas.get(), weights, params);

            // portfolio eval
            evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_portfolio_values.get(), params);

            // copy terminal portfolio values to compute VaR threshold on host
            CUDA_CHECK(cudaMemcpy(h_portfolio_values.data(), d_portfolio_values.get(), params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));

            // calculate 5th percentile VaR cutoff on host
            std::vector<float> sorted_v = h_portfolio_values;
            size_t tail_index = static_cast<size_t>(params.num_paths * (1.0f - cvar_alpha));
            std::nth_element(sorted_v.begin(), sorted_v.begin() + tail_index, sorted_v.end());
            float var_cutoff = sorted_v[tail_index];

            // backprop through time with Hybrid Loss
            bptt_backward_pass(
                d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_portfolio_values.get(), 
                weights, grads, params, var_cutoff, cvar_alpha, lambda_cvar, use_hybrid
            );

            // adam optim step
            adam_step(weights, grads, adam_state, adam_params);
            adam_params.time_step++;

            if(epoch % 10 == 0 || epoch == num_epochs){
                // compute MSE component
                double sum_sq_error = 0.0;
                for (float v : h_portfolio_values) sum_sq_error += (v * v);
                double mse_loss = sum_sq_error / params.num_paths;

                // compute CVaR component
                double cvar_sum = 0.0;
                int tail_count = 0;
                for (float v : h_portfolio_values) {
                    if (v <= var_cutoff) {
                        cvar_sum += (-v);
                        tail_count++;
                    }
                }
                double cvar_loss = (tail_count > 0) ? (cvar_sum / tail_count) : 0.0;
                double total_hybrid_loss = mse_loss + lambda_cvar * cvar_loss;

                // save weight checkpoint if new best hybrid loss achieved
                if (total_hybrid_loss < best_loss) {
                    best_loss = total_hybrid_loss;
                    best_epoch = epoch;

                    CUDA_CHECK(cudaMemcpy(best_W1.data(), d_W1.get(), best_W1.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_W2.data(), d_W2.get(), best_W2.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_W3.data(), d_W3.get(), best_W3.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_b1.data(), d_b1.get(), best_b1.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_b2.data(), d_b2.get(), best_b2.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_b3.data(), d_b3.get(), best_b3.size() * sizeof(float), cudaMemcpyDeviceToHost));
                }

                std::cout << "epoch [" << epoch << "/" << num_epochs << "] hybrid loss: " << total_hybrid_loss
                          << " (MSE: " << mse_loss << ", CVaR: " << cvar_loss << ") best: " << best_loss << " @ epoch " << best_epoch << std::endl;
            }
        }

        // restore best weights to device
        std::cout << "\nrestoring best weights from epoch " << best_epoch << " (best training hybrid loss: " << best_loss << ")..." << std::endl;
        CUDA_CHECK(cudaMemcpy(d_W1.get(), best_W1.data(), best_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1.get(), best_b1.data(), best_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2.get(), best_W2.data(), best_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2.get(), best_b2.data(), best_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W3.get(), best_W3.data(), best_W3.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3.get(), best_b3.data(), best_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

        export_hedging_surface("../results/hedging_surface.csv", weights, params);

        // out-of-sample testing
        std::cout << "\n--- running out-of-sample validation on unseen real snp500 test paths ---" << std::endl;
        RealDataset test_ds = load_and_normalize_real_csv("../data/GSPC_test_paths.csv", params);

        SimulationParams test_params = params;
        test_params.num_paths = test_ds.num_paths;

        size_t test_elements = test_params.num_paths * test_params.num_steps;
        auto d_test_paths = cuda_utils::make_device_buffer<float>(test_elements);
        auto d_test_payoffs = cuda_utils::make_device_buffer<float>(test_params.num_paths);
        auto d_test_bs_deltas = cuda_utils::make_device_buffer<float>(test_elements);
        auto d_test_policy_deltas = cuda_utils::make_device_buffer<float>(test_elements);
        auto d_test_portfolio_bs = cuda_utils::make_device_buffer<float>(test_params.num_paths);
        auto d_test_portfolio_policy = cuda_utils::make_device_buffer<float>(test_params.num_paths);

        auto d_test_traj_bs = cuda_utils::make_device_buffer<float>(test_elements);
        auto d_test_traj_policy = cuda_utils::make_device_buffer<float>(test_elements);

        CUDA_CHECK(cudaMemcpy(d_test_paths.get(), test_ds.h_paths.data(), test_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_test_payoffs.get(), test_ds.h_payoffs.data(), test_params.num_paths * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_test_bs_deltas.get(), test_ds.h_deltas.data(), test_elements * sizeof(float), cudaMemcpyHostToDevice));

        forward_pass(d_test_paths.get(), d_test_policy_deltas.get(), weights, test_params);

        evaluate_portfolio(d_test_paths.get(), d_test_payoffs.get(), d_test_bs_deltas.get(), d_test_portfolio_bs.get(), test_params);
        evaluate_portfolio(d_test_paths.get(), d_test_payoffs.get(), d_test_policy_deltas.get(), d_test_portfolio_policy.get(), test_params);

        evaluate_portfolio_trajectories(d_test_paths.get(), d_test_payoffs.get(), d_test_bs_deltas.get(), d_test_traj_bs.get(), test_params);
        evaluate_portfolio_trajectories(d_test_paths.get(), d_test_payoffs.get(), d_test_policy_deltas.get(), d_test_traj_policy.get(), test_params);

        std::vector<float> h_test_bs(test_params.num_paths);
        std::vector<float> h_test_policy(test_params.num_paths);
        CUDA_CHECK(cudaMemcpy(h_test_bs.data(), d_test_portfolio_bs.get(), test_params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_test_policy.data(), d_test_portfolio_policy.get(), test_params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));

        // write terminal pnls
        std::ofstream pnl_csv("../results/test_pnl_distribution.csv");
        if (!pnl_csv.is_open()) {
            throw std::runtime_error("failed to open test_pnl_distribution.csv for writing");
        }
        pnl_csv << "BS_PnL,Policy_PnL\n";
        for (int i = 0; i < test_params.num_paths; ++i) {
            pnl_csv << h_test_bs[i] << "," << h_test_policy[i] << "\n";
        }
        pnl_csv.close();
        std::cout << "exported snp500 test PnLs to test_pnl_distribution.csv" << std::endl;

        std::vector<float> h_test_traj_bs(test_elements);
        std::vector<float> h_test_traj_policy(test_elements);
        CUDA_CHECK(cudaMemcpy(h_test_traj_bs.data(), d_test_traj_bs.get(), test_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_test_traj_policy.data(), d_test_traj_policy.get(), test_elements * sizeof(float), cudaMemcpyDeviceToHost));

        // write mean trajectories
        std::ofstream traj_csv("../results/test_equity_trajectories.csv");
        if (!traj_csv.is_open()) {
            throw std::runtime_error("failed to open test_equity_trajectories.csv for writing");
        }
        traj_csv << "step,bs_mean,bs_std,policy_mean,policy_std\n";
        for (int t = 0; t < test_params.num_steps; ++t) {
            double sum_bs = 0.0, sum_policy = 0.0;
            for (int i = 0; i < test_params.num_paths; ++i) {
                sum_bs += h_test_traj_bs[i * test_params.num_steps + t];
                sum_policy += h_test_traj_policy[i * test_params.num_steps + t];
            }
            double mean_bs = sum_bs / test_params.num_paths;
            double mean_policy = sum_policy / test_params.num_paths;

            double var_bs = 0.0, var_policy = 0.0;
            for (int i = 0; i < test_params.num_paths; ++i) {
                double diff_bs = h_test_traj_bs[i * test_params.num_steps + t] - mean_bs;
                double diff_policy = h_test_traj_policy[i * test_params.num_steps + t] - mean_policy;
                var_bs += diff_bs * diff_bs;
                var_policy += diff_policy * diff_policy;
            }
            double std_bs = std::sqrt(var_bs / test_params.num_paths);
            double std_policy = std::sqrt(var_policy / test_params.num_paths);

            traj_csv << t << "," << mean_bs << "," << std_bs << "," << mean_policy << "," << std_policy << "\n";
        }
        traj_csv.close();
        std::cout << "exported snp500 equity trajectories to test_equity_trajectories.csv" << std::endl;

        // write sample individual path trajectories
        std::ofstream sample_csv("../results/sample_equity_paths.csv");
        if (!sample_csv.is_open()) {
            throw std::runtime_error("failed to open sample_equity_paths.csv for writing");
        }
        sample_csv << "path_idx,step,bs_pnl,policy_pnl\n";
        int num_sample_paths = 20;
        for (int i = 0; i < num_sample_paths; ++i) {
            for (int t = 0; t < test_params.num_steps; ++t) {
                int idx = i * test_params.num_steps + t;
                sample_csv << i << "," << t << "," << h_test_traj_bs[idx] << "," << h_test_traj_policy[idx] << "\n";
            }
        }
        sample_csv.close();
        std::cout << "exported 20 sample path equity trajectories to sample_equity_paths.csv" << std::endl;

        std::cout << "done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}