#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <string>

#include "simulation.cuh"
#include "mlp.cuh"
#include "portfolio.cuh"
#include "autograd.cuh"
#include "optimizer.cuh"
#include "exporter.cuh"

namespace {

float analytical_bs_call(float S0, float K, float r, float sigma, float T) {
    double sqrt_T = std::sqrt(static_cast<double>(T));
    double d1 = (std::log(static_cast<double>(S0) / static_cast<double>(K))
                 + (static_cast<double>(r) + 0.5 * static_cast<double>(sigma) * static_cast<double>(sigma))
                       * static_cast<double>(T))
                / (static_cast<double>(sigma) * sqrt_T);
    double d2 = d1 - static_cast<double>(sigma) * sqrt_T;
    auto norm_cdf = [](double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); };
    return static_cast<float>(
        static_cast<double>(S0) * norm_cdf(d1)
        - static_cast<double>(K) * std::exp(-static_cast<double>(r) * static_cast<double>(T)) * norm_cdf(d2)
    );
}

struct DeviceDatasetBuffers {
    SimulationParams params;
    cuda_utils::device_ptr<float> d_paths;
    cuda_utils::device_ptr<float> d_vols;
    cuda_utils::device_ptr<float> d_payoffs;
    cuda_utils::device_ptr<float> d_bs_deltas;
    cuda_utils::device_ptr<float> d_policy_deltas;
    cuda_utils::device_ptr<float> d_portfolio;
};

DeviceDatasetBuffers generate_sim_dataset(const SimulationParams& base_params, int num_paths, unsigned long long seed) {
    DeviceDatasetBuffers buf;
    buf.params = base_params;
    buf.params.num_paths = num_paths;

    size_t total_elements = static_cast<size_t>(num_paths) * static_cast<size_t>(base_params.num_steps);
    buf.d_paths = cuda_utils::make_device_buffer<float>(total_elements);
    buf.d_vols = cuda_utils::make_device_buffer<float>(total_elements);
    buf.d_payoffs = cuda_utils::make_device_buffer<float>(num_paths);
    buf.d_bs_deltas = cuda_utils::make_device_buffer<float>(total_elements);
    buf.d_policy_deltas = cuda_utils::make_device_buffer<float>(total_elements);
    buf.d_portfolio = cuda_utils::make_device_buffer<float>(num_paths);

    generate_market_paths(
        buf.d_paths.get(), buf.d_vols.get(), buf.d_payoffs.get(), buf.d_bs_deltas.get(),
        buf.params, seed
    );
    return buf;
}

double compute_mean_turnover(const std::vector<float>& deltas, int num_paths, int num_steps) {
    if (num_paths <= 0 || num_steps <= 1) return 0.0;
    double total_turnover = 0.0;
    for (int i = 0; i < num_paths; ++i) {
        double prev = 0.0;
        for (int t = 0; t < num_steps - 1; ++t) {
            double curr = static_cast<double>(deltas[static_cast<size_t>(t) * num_paths + i]);
            total_turnover += std::fabs(curr - prev);
            prev = curr;
        }
        // include expiry liquidation to match portfolio settlement
        total_turnover += std::fabs(prev);
    }
    return total_turnover / static_cast<double>(num_paths);
}

} // namespace

int main(int argc, char** argv) {
    try {
        MarketModel model = MarketModel::GBM;
        int train_paths = 131072;
        int val_paths = 32768;
        int test_paths = 100000;
        int num_epochs = 400;
        unsigned long long base_seed = 42ULL;

        float custom_cost_ratio = 0.0010f;
        HedgingLossType loss_type = HedgingLossType::MeanVarCvar;
        float risk_aversion = 2.0f;
        float lambda_cvar = 0.25f;
        float weight_decay = 1e-4f;
        float initial_lr = 0.0020f;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                std::string m = argv[++i];
                model = (m == "heston") ? MarketModel::Heston : MarketModel::GBM;
            } else if (arg == "--train-paths" && i + 1 < argc) {
                train_paths = std::stoi(argv[++i]);
            } else if (arg == "--val-paths" && i + 1 < argc) {
                val_paths = std::stoi(argv[++i]);
            } else if (arg == "--test-paths" && i + 1 < argc) {
                test_paths = std::stoi(argv[++i]);
            } else if (arg == "--epochs" && i + 1 < argc) {
                num_epochs = std::stoi(argv[++i]);
            } else if (arg == "--seed" && i + 1 < argc) {
                base_seed = std::stoull(argv[++i]);
            } else if (arg == "--cost" && i + 1 < argc) {
                custom_cost_ratio = std::stof(argv[++i]);
            } else if (arg == "--loss" && i + 1 < argc) {
                std::string lstr = argv[++i];
                if (lstr == "centered") {
                    loss_type = HedgingLossType::CenteredVarCvar;
                } else if (lstr == "entropic") {
                    loss_type = HedgingLossType::EntropicRisk;
                } else {
                    loss_type = HedgingLossType::MeanVarCvar;
                }
            } else if (arg == "--gamma" && i + 1 < argc) {
                risk_aversion = std::stof(argv[++i]);
            } else if (arg == "--lambda" && i + 1 < argc) {
                lambda_cvar = std::stof(argv[++i]);
            } else if (arg == "--wd" && i + 1 < argc) {
                weight_decay = std::stof(argv[++i]);
            } else if (arg == "--lr" && i + 1 < argc) {
                initial_lr = std::stof(argv[++i]);
            }
        }

        std::filesystem::create_directories("../results");

        size_t free_initial = 0, total_vram = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_initial, &total_vram));

        SimulationParams params{};
        params.num_steps = 30;
        params.S0 = 100.0f;
        params.K = 100.0f;
        params.r = 0.05f;
        params.T = 30.0f / 252.0f;
        params.dt = params.T / static_cast<float>(params.num_steps - 1);
        params.cost_ratio = custom_cost_ratio;
        params.model = model;
        params.v0 = 0.04f;
        params.theta = 0.04f;
        params.kappa = 2.0f;
        params.xi = 0.50f;
        params.rho = -0.70f;
        params.heston_substeps = 8;
        params.sigma = (model == MarketModel::Heston) ? std::sqrt(params.theta) : 0.20f;

        if (model == MarketModel::Heston) {
            params.option_price = estimate_mc_option_price(params, 200000, base_seed + 7919ULL);
        } else {
            params.option_price = analytical_bs_call(params.S0, params.K, params.r, params.sigma, params.T);
        }

        const char* model_name = (model == MarketModel::Heston) ? "heston" : "gbm";
        std::cout << "model=" << model_name
                  << ", sigma=" << params.sigma
                  << ", T=" << params.T << " yr"
                  << ", cost=" << params.cost_ratio * 10000.0f << " bps"
                  << ", C0=" << params.option_price << std::endl;

        DeviceDatasetBuffers train_buf = generate_sim_dataset(params, train_paths, base_seed);
        DeviceDatasetBuffers val_buf = generate_sim_dataset(params, val_paths, base_seed + 100003ULL);
        DeviceDatasetBuffers test_buf = generate_sim_dataset(params, test_paths, base_seed + 200003ULL);

        params.num_paths = train_buf.params.num_paths;

        auto d_W1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_b1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_W2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_b2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_W3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_b3 = cuda_utils::make_device_buffer<float>(output_dim);

        auto d_gW1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_gb1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_gW2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_gb2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_gW3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_gb3 = cuda_utils::make_device_buffer<float>(output_dim);

        auto d_mW1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_mb1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_mW2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_mb2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_mW3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_mb3 = cuda_utils::make_device_buffer<float>(output_dim);

        auto d_vW1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_vb1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_vW2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_vb2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_vW3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_vb3 = cuda_utils::make_device_buffer<float>(output_dim);

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

        std::mt19937 rng(static_cast<unsigned int>(base_seed));
        std::vector<float> h_W1(hidden_dim * input_dim);
        std::vector<float> h_b1(hidden_dim, 0.0f);
        std::vector<float> h_W2(hidden_dim * hidden_dim);
        std::vector<float> h_b2(hidden_dim, 0.0f);
        std::vector<float> h_W3(hidden_dim * output_dim);
        std::vector<float> h_b3(output_dim, 0.0f);

        auto fill_xavier = [&](std::vector<float>& vec, float std_dev) {
            std::normal_distribution<float> dist(0.0f, std_dev);
            for (auto& val : vec) val = dist(rng);
        };

        fill_xavier(h_W1, std::sqrt(2.0f / (input_dim + hidden_dim)));
        fill_xavier(h_W2, std::sqrt(2.0f / (hidden_dim + hidden_dim)));
        fill_xavier(h_W3, std::sqrt(2.0f / (hidden_dim + output_dim)));

        std::vector<float> best_W1 = h_W1;
        std::vector<float> best_b1 = h_b1;
        std::vector<float> best_W2 = h_W2;
        std::vector<float> best_b2 = h_b2;
        std::vector<float> best_W3 = h_W3;
        std::vector<float> best_b3 = h_b3;

        CUDA_CHECK(cudaMemcpy(d_W1.get(), h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1.get(), h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2.get(), h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2.get(), h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W3.get(), h_W3.data(), h_W3.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3.get(), h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

        MLPWeights weights{d_W1.get(), d_b1.get(), d_W2.get(), d_b2.get(), d_W3.get(), d_b3.get()};
        MLPGradients grads{d_gW1.get(), d_gb1.get(), d_gW2.get(), d_gb2.get(), d_gW3.get(), d_gb3.get()};
        AdamState adam_state{
            d_mW1.get(), d_mb1.get(), d_mW2.get(), d_mb2.get(), d_mW3.get(), d_mb3.get(),
            d_vW1.get(), d_vb1.get(), d_vW2.get(), d_vb2.get(), d_vW3.get(), d_vb3.get()
        };

        AdamParams adam_params;
        adam_params.weight_decay = weight_decay;
        float min_lr = 0.00005f;

        float cvar_alpha = 0.95f;
        float cvar_eps = 0.02f;
        bool use_hybrid = true;

        size_t free_after_alloc = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_after_alloc, &total_vram));
        double peak_vram_mb = static_cast<double>(free_initial - std::min(free_initial, free_after_alloc)) / (1024.0 * 1024.0);
        double total_vram_mb = static_cast<double>(total_vram) / (1024.0 * 1024.0);
        std::cout << "allocated train=" << train_paths << ", val=" << val_paths << ", test=" << test_paths
                  << " | VRAM delta=" << peak_vram_mb << " MB (total GPU " << total_vram_mb << " MB)" << std::endl;

        evaluate_portfolio(
            train_buf.d_paths.get(), train_buf.d_payoffs.get(), train_buf.d_bs_deltas.get(),
            train_buf.d_portfolio.get(), train_buf.params
        );
        std::vector<float> h_train_portfolio(train_buf.params.num_paths);
        CUDA_CHECK(cudaMemcpy(h_train_portfolio.data(), train_buf.d_portfolio.get(),
                              train_buf.params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        HedgingLossStats bs_train_stats = compute_hedging_loss_stats(
            h_train_portfolio, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
        );

        evaluate_portfolio(
            val_buf.d_paths.get(), val_buf.d_payoffs.get(), val_buf.d_bs_deltas.get(),
            val_buf.d_portfolio.get(), val_buf.params
        );
        std::vector<float> h_val_portfolio(val_buf.params.num_paths);
        CUDA_CHECK(cudaMemcpy(h_val_portfolio.data(), val_buf.d_portfolio.get(),
                              val_buf.params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        HedgingLossStats bs_val_stats = compute_hedging_loss_stats(
            h_val_portfolio, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
        );

        std::cout << "BS baseline (" << params.cost_ratio * 10000.0f << " bps fee) | "
                  << "train var=" << bs_train_stats.variance << ", loss=" << bs_train_stats.total_loss
                  << " | val var=" << bs_val_stats.variance << ", loss=" << bs_val_stats.total_loss << std::endl;

        std::ofstream loss_csv("../results/training_losses.csv");
        if (!loss_csv.is_open()) {
            throw std::runtime_error("failed to open training_losses.csv for writing");
        }
        loss_csv << "epoch,train_var,train_cvar,train_loss,val_var,val_cvar,val_loss\n";

        init_optimizer_weights(weights);

        if (loss_type == HedgingLossType::EntropicRisk) {
            std::cout << "training MLP with EntropicRisk (gamma=" << risk_aversion
                      << ", wd=" << weight_decay << ")..." << std::endl;
        } else if (loss_type == HedgingLossType::MeanVarCvar) {
            std::cout << "training MLP with MeanVarCvar (Var + " << lambda_cvar
                      << " * smooth RU-CVaR@" << cvar_alpha * 100 << "%, wd=" << weight_decay << ")..." << std::endl;
        } else {
            std::cout << "training MLP with CenteredVarCvar (Var + " << lambda_cvar
                      << " * smooth RU-CVaR@" << cvar_alpha * 100 << "%, wd=" << weight_decay << ")..." << std::endl;
        }

        double best_val_loss = std::numeric_limits<double>::max();
        int best_epoch = 0;

        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start));

        for (int epoch = 1; epoch <= num_epochs; ++epoch) {
            adam_params.lr = min_lr + 0.5f * (initial_lr - min_lr)
                             * (1.0f + std::cos(static_cast<float>(M_PI) * epoch / num_epochs));

            forward_pass(train_buf.d_paths.get(), train_buf.d_vols.get(), train_buf.d_policy_deltas.get(), weights, train_buf.params);
            evaluate_portfolio(
                train_buf.d_paths.get(), train_buf.d_payoffs.get(), train_buf.d_policy_deltas.get(),
                train_buf.d_portfolio.get(), train_buf.params
            );

            CUDA_CHECK(cudaMemcpy(h_train_portfolio.data(), train_buf.d_portfolio.get(),
                                  train_buf.params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
            HedgingLossStats train_stats = compute_hedging_loss_stats(
                h_train_portfolio, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
            );

            bptt_backward_pass(
                train_buf.d_paths.get(), train_buf.d_vols.get(), train_buf.d_payoffs.get(),
                train_buf.d_policy_deltas.get(), train_buf.d_portfolio.get(),
                weights, grads, train_buf.params,
                train_stats, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
            );

            adam_step(weights, grads, adam_state, adam_params);
            adam_params.time_step++;

            if (epoch % 10 == 0 || epoch == num_epochs) {
                forward_pass(val_buf.d_paths.get(), val_buf.d_vols.get(), val_buf.d_policy_deltas.get(), weights, val_buf.params);
                evaluate_portfolio(
                    val_buf.d_paths.get(), val_buf.d_payoffs.get(), val_buf.d_policy_deltas.get(),
                    val_buf.d_portfolio.get(), val_buf.params
                );
                CUDA_CHECK(cudaMemcpy(h_val_portfolio.data(), val_buf.d_portfolio.get(),
                                      val_buf.params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
                HedgingLossStats val_stats = compute_hedging_loss_stats(
                    h_val_portfolio, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
                );

                loss_csv << epoch << "," << train_stats.variance << "," << train_stats.ru_cvar << ","
                         << train_stats.total_loss << "," << val_stats.variance << ","
                         << val_stats.ru_cvar << "," << val_stats.total_loss << "\n";
                loss_csv.flush();

                if (val_stats.total_loss < best_val_loss) {
                    best_val_loss = val_stats.total_loss;
                    best_epoch = epoch;

                    CUDA_CHECK(cudaMemcpy(best_W1.data(), d_W1.get(), best_W1.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_W2.data(), d_W2.get(), best_W2.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_W3.data(), d_W3.get(), best_W3.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_b1.data(), d_b1.get(), best_b1.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_b2.data(), d_b2.get(), best_b2.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_b3.data(), d_b3.get(), best_b3.size() * sizeof(float), cudaMemcpyDeviceToHost));
                }

                if (epoch % 50 == 0 || epoch == 10 || epoch == num_epochs) {
                    std::cout << "epoch [" << epoch << "/" << num_epochs << "] "
                              << "train loss=" << train_stats.total_loss << " (var=" << train_stats.variance << ") | "
                              << "val loss=" << val_stats.total_loss << " (var=" << val_stats.variance << ") "
                              << "best_val=" << best_val_loss << " @ epoch " << best_epoch << std::endl;
                }
            }
        }

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        double ms_per_epoch = static_cast<double>(elapsed_ms) / static_cast<double>(num_epochs);
        double steps_per_sec = (static_cast<double>(train_paths) * static_cast<double>(params.num_steps - 1) * static_cast<double>(num_epochs))
                               / (static_cast<double>(elapsed_ms) / 1000.0);
        std::cout << "\ntraining throughput: " << ms_per_epoch << " ms/epoch, "
                  << (steps_per_sec / 1e6) << " M paths*steps/sec" << std::endl;

        std::cout << "restoring best weights from epoch " << best_epoch
                  << " (best validation loss: " << best_val_loss << ")..." << std::endl;
        CUDA_CHECK(cudaMemcpy(d_W1.get(), best_W1.data(), best_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1.get(), best_b1.data(), best_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2.get(), best_W2.data(), best_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2.get(), best_b2.data(), best_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W3.get(), best_W3.data(), best_W3.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3.get(), best_b3.data(), best_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

        export_hedging_surface("../results/hedging_surface.csv", weights, params);

        std::cout << "\n--- out-of-sample evaluation (" << test_paths << " " << model_name << " paths) ---" << std::endl;

        size_t test_elements = static_cast<size_t>(test_buf.params.num_paths) * static_cast<size_t>(test_buf.params.num_steps);
        auto d_test_portfolio_bs = cuda_utils::make_device_buffer<float>(test_buf.params.num_paths);
        auto d_test_traj_bs = cuda_utils::make_device_buffer<float>(test_elements);
        auto d_test_traj_policy = cuda_utils::make_device_buffer<float>(test_elements);

        forward_pass(test_buf.d_paths.get(), test_buf.d_vols.get(), test_buf.d_policy_deltas.get(), weights, test_buf.params);

        evaluate_portfolio(
            test_buf.d_paths.get(), test_buf.d_payoffs.get(), test_buf.d_bs_deltas.get(),
            d_test_portfolio_bs.get(), test_buf.params
        );
        evaluate_portfolio(
            test_buf.d_paths.get(), test_buf.d_payoffs.get(), test_buf.d_policy_deltas.get(),
            test_buf.d_portfolio.get(), test_buf.params
        );

        evaluate_portfolio_trajectories(
            test_buf.d_paths.get(), test_buf.d_payoffs.get(), test_buf.d_bs_deltas.get(),
            d_test_traj_bs.get(), test_buf.params
        );
        evaluate_portfolio_trajectories(
            test_buf.d_paths.get(), test_buf.d_payoffs.get(), test_buf.d_policy_deltas.get(),
            d_test_traj_policy.get(), test_buf.params
        );

        std::vector<float> h_test_bs(test_buf.params.num_paths);
        std::vector<float> h_test_policy(test_buf.params.num_paths);
        std::vector<float> h_test_bs_deltas(test_elements);
        std::vector<float> h_test_policy_deltas(test_elements);

        CUDA_CHECK(cudaMemcpy(h_test_bs.data(), d_test_portfolio_bs.get(),
                              test_buf.params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_test_policy.data(), test_buf.d_portfolio.get(),
                              test_buf.params.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_test_bs_deltas.data(), test_buf.d_bs_deltas.get(),
                              test_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_test_policy_deltas.data(), test_buf.d_policy_deltas.get(),
                              test_elements * sizeof(float), cudaMemcpyDeviceToHost));

        HedgingLossStats bs_test_stats = compute_hedging_loss_stats(
            h_test_bs, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, true
        );
        HedgingLossStats policy_test_stats = compute_hedging_loss_stats(
            h_test_policy, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, true
        );

        double bs_turnover = compute_mean_turnover(h_test_bs_deltas, test_buf.params.num_paths, test_buf.params.num_steps);
        double policy_turnover = compute_mean_turnover(h_test_policy_deltas, test_buf.params.num_paths, test_buf.params.num_steps);

        std::cout << "test BS     | mean=" << bs_test_stats.mean_v
                  << ", std=" << std::sqrt(bs_test_stats.variance)
                  << ", var=" << bs_test_stats.variance
                  << ", cvar95=" << bs_test_stats.ru_cvar
                  << ", turnover=" << bs_turnover
                  << ", loss=" << bs_test_stats.total_loss << std::endl;
        std::cout << "test Policy | mean=" << policy_test_stats.mean_v
                  << ", std=" << std::sqrt(policy_test_stats.variance)
                  << ", var=" << policy_test_stats.variance
                  << ", cvar95=" << policy_test_stats.ru_cvar
                  << ", turnover=" << policy_turnover
                  << ", loss=" << policy_test_stats.total_loss << std::endl;

        std::ofstream meta_csv("../results/run_metadata.csv");
        if (!meta_csv.is_open()) {
            throw std::runtime_error("failed to open run_metadata.csv for writing");
        }
        meta_csv << "model,num_test_paths,num_steps,S0,r,sigma,T,dt,cost_ratio,option_price,best_epoch,bs_turnover,policy_turnover,peak_vram_mb,ms_per_epoch,msteps_per_sec\n";
        meta_csv << model_name << ","
                 << test_buf.params.num_paths << ","
                 << test_buf.params.num_steps << ","
                 << test_buf.params.S0 << ","
                 << test_buf.params.r << ","
                 << test_buf.params.sigma << ","
                 << test_buf.params.T << ","
                 << test_buf.params.dt << ","
                 << test_buf.params.cost_ratio << ","
                 << test_buf.params.option_price << ","
                 << best_epoch << ","
                 << bs_turnover << ","
                 << policy_turnover << ","
                 << peak_vram_mb << ","
                 << ms_per_epoch << ","
                 << (steps_per_sec / 1e6) << "\n";
        meta_csv.close();

        std::ofstream pnl_csv("../results/test_pnl_distribution.csv");
        if (!pnl_csv.is_open()) {
            throw std::runtime_error("failed to open test_pnl_distribution.csv for writing");
        }
        pnl_csv << "BS_PnL,Policy_PnL\n";
        for (int i = 0; i < test_buf.params.num_paths; ++i) {
            pnl_csv << h_test_bs[i] << "," << h_test_policy[i] << "\n";
        }
        pnl_csv.close();

        std::vector<float> h_test_traj_bs(test_elements);
        std::vector<float> h_test_traj_policy(test_elements);
        CUDA_CHECK(cudaMemcpy(h_test_traj_bs.data(), d_test_traj_bs.get(), test_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_test_traj_policy.data(), d_test_traj_policy.get(), test_elements * sizeof(float), cudaMemcpyDeviceToHost));

        std::ofstream traj_csv("../results/test_equity_trajectories.csv");
        if (!traj_csv.is_open()) {
            throw std::runtime_error("failed to open test_equity_trajectories.csv for writing");
        }
        traj_csv << "step,bs_mean,bs_std,policy_mean,policy_std\n";
        for (int t = 0; t < test_buf.params.num_steps; ++t) {
            double sum_bs = 0.0, sum_policy = 0.0;
            for (int i = 0; i < test_buf.params.num_paths; ++i) {
                sum_bs += h_test_traj_bs[static_cast<size_t>(i) * test_buf.params.num_steps + t];
                sum_policy += h_test_traj_policy[static_cast<size_t>(i) * test_buf.params.num_steps + t];
            }
            double mean_bs = sum_bs / test_buf.params.num_paths;
            double mean_policy = sum_policy / test_buf.params.num_paths;

            double var_bs = 0.0, var_policy = 0.0;
            for (int i = 0; i < test_buf.params.num_paths; ++i) {
                double diff_bs = h_test_traj_bs[static_cast<size_t>(i) * test_buf.params.num_steps + t] - mean_bs;
                double diff_policy = h_test_traj_policy[static_cast<size_t>(i) * test_buf.params.num_steps + t] - mean_policy;
                var_bs += diff_bs * diff_bs;
                var_policy += diff_policy * diff_policy;
            }
            double std_bs = std::sqrt(var_bs / test_buf.params.num_paths);
            double std_policy = std::sqrt(var_policy / test_buf.params.num_paths);

            traj_csv << t << "," << mean_bs << "," << std_bs << "," << mean_policy << "," << std_policy << "\n";
        }
        traj_csv.close();

        std::cout << "done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
