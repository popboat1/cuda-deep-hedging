#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

#include "cuda_utils.cuh"
#include "simulation.cuh"
#include "mlp.cuh"
#include "portfolio.cuh"
#include "autograd.cuh"
#include "optimizer.cuh"

static float bs_call_price(float S, float K, float r, float sigma, float T) {
    float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * std::sqrt(T));
    float d2 = d1 - sigma * std::sqrt(T);
    auto ncdf = [](float x) { return 0.5f * std::erfc(-x * static_cast<float>(M_SQRT1_2)); };
    return S * ncdf(d1) - K * std::exp(-r * T) * ncdf(d2);
}

static double evaluate_loss_for_weights(
    const float* d_paths,
    const float* d_vols,
    const float* d_payoffs,
    float* d_policy_deltas,
    float* d_v,
    const MLPWeights& weights,
    const SimulationParams& p,
    float cvar_alpha,
    float cvar_eps,
    float lambda_cvar,
    bool use_hybrid,
    HedgingLossType loss_type,
    float risk_aversion
) {
    forward_pass(d_paths, d_vols, d_policy_deltas, weights, p);
    evaluate_portfolio(d_paths, d_payoffs, d_policy_deltas, d_v, p);
    std::vector<float> h_v(p.num_paths);
    CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, p.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
    return compute_hedging_loss_stats(
        h_v, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
    ).total_loss;
}

static int check_finite_diff_grad(
    float* d_weight_buf,
    int param_idx,
    const float* d_grad_buf,
    const float* d_paths,
    const float* d_vols,
    const float* d_payoffs,
    float* d_policy_deltas,
    float* d_v,
    const MLPWeights& weights,
    const SimulationParams& p,
    float cvar_alpha,
    float cvar_eps,
    float lambda_cvar,
    bool use_hybrid,
    HedgingLossType loss_type,
    float risk_aversion,
    const char* name
) {
    float orig_val = 0.0f;
    CUDA_CHECK(cudaMemcpy(&orig_val, d_weight_buf + param_idx, sizeof(float), cudaMemcpyDeviceToHost));

    float anal_grad = 0.0f;
    CUDA_CHECK(cudaMemcpy(&anal_grad, d_grad_buf + param_idx, sizeof(float), cudaMemcpyDeviceToHost));

    float eps = 5e-3f;
    float plus_val = orig_val + eps;
    float minus_val = orig_val - eps;

    CUDA_CHECK(cudaMemcpy(d_weight_buf + param_idx, &plus_val, sizeof(float), cudaMemcpyHostToDevice));
    double loss_plus = evaluate_loss_for_weights(
        d_paths, d_vols, d_payoffs, d_policy_deltas, d_v, weights, p,
        cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
    );

    CUDA_CHECK(cudaMemcpy(d_weight_buf + param_idx, &minus_val, sizeof(float), cudaMemcpyHostToDevice));
    double loss_minus = evaluate_loss_for_weights(
        d_paths, d_vols, d_payoffs, d_policy_deltas, d_v, weights, p,
        cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
    );

    CUDA_CHECK(cudaMemcpy(d_weight_buf + param_idx, &orig_val, sizeof(float), cudaMemcpyHostToDevice));

    double fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double abs_err = std::fabs(fd_grad - static_cast<double>(anal_grad));
    double denom = std::max(1e-2, std::max(std::fabs(fd_grad), std::fabs(static_cast<double>(anal_grad))));
    double rel_err = abs_err / denom;

    if (rel_err > 0.03 && abs_err > 5e-4) {
        std::cerr << "FAIL [fd_grad " << name << " idx=" << param_idx << "]: anal=" << anal_grad
                  << ", fd=" << fd_grad << ", abs_err=" << abs_err << "\n";
        return 1;
    }
    return 0;
}

int main() {
    int failures = 0;

    // test 1: exact 29-interval accrual, consistency, no expiry rebalance, and zero Day-0 MTM gap
    {
        SimulationParams p{};
        p.num_paths = 4;
        p.num_steps = 30;
        p.S0 = 100.0f;
        p.K = 100.0f;
        p.r = 0.05f;
        p.sigma = 0.20f;
        p.T = 0.5f;
        p.dt = p.T / (p.num_steps - 1);
        p.cost_ratio = 0.001f;
        p.option_price = 10.0f;

        std::vector<float> h_paths(p.num_paths * p.num_steps, 100.0f);
        std::vector<float> h_payoffs(p.num_paths, 0.0f);
        std::vector<float> h_deltas(p.num_paths * p.num_steps, 0.0f);

        auto d_paths = cuda_utils::make_device_buffer<float>(h_paths.size());
        auto d_payoffs = cuda_utils::make_device_buffer<float>(h_payoffs.size());
        auto d_deltas = cuda_utils::make_device_buffer<float>(h_deltas.size());
        auto d_v = cuda_utils::make_device_buffer<float>(p.num_paths);
        auto d_traj = cuda_utils::make_device_buffer<float>(h_paths.size());

        CUDA_CHECK(cudaMemcpy(d_paths.get(), h_paths.data(), h_paths.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_payoffs.get(), h_payoffs.data(), h_payoffs.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_deltas.get(), h_deltas.data(), h_deltas.size() * sizeof(float), cudaMemcpyHostToDevice));

        evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_deltas.get(), d_v.get(), p);
        evaluate_portfolio_trajectories(d_paths.get(), d_payoffs.get(), d_deltas.get(), d_traj.get(), p);

        std::vector<float> h_v(p.num_paths);
        std::vector<float> h_traj(h_paths.size());
        CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), p.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_traj.data(), d_traj.get(), h_paths.size() * sizeof(float), cudaMemcpyDeviceToHost));

        float expected_v = 10.0f * std::exp(p.r * p.T);
        if (std::fabs(h_v[0] - expected_v) > 1e-4f) {
            std::cerr << "FAIL [accrual]: got " << h_v[0] << ", expected " << expected_v << "\n";
            failures++;
        }

        p.option_price = bs_call_price(p.S0, p.K, p.r, p.sigma, p.T);
        std::fill(h_deltas.begin(), h_deltas.end(), 0.5f);
        CUDA_CHECK(cudaMemcpy(d_deltas.get(), h_deltas.data(), h_deltas.size() * sizeof(float), cudaMemcpyHostToDevice));
        evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_deltas.get(), d_v.get(), p);
        evaluate_portfolio_trajectories(d_paths.get(), d_payoffs.get(), d_deltas.get(), d_traj.get(), p);
        CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), p.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_traj.data(), d_traj.get(), h_paths.size() * sizeof(float), cudaMemcpyDeviceToHost));

        float traj_last = h_traj[0 * p.num_steps + (p.num_steps - 1)];
        if (std::fabs(h_v[0] - traj_last) > 1e-4f) {
            std::cerr << "FAIL [consistency]: eval_portfolio=" << h_v[0] << ", traj_last=" << traj_last << "\n";
            failures++;
        }

        float expected_day0 = -p.cost_ratio * 0.5f * p.S0;
        if (std::fabs(h_traj[0] - expected_day0) > 1e-4f) {
            std::cerr << "FAIL [day0_mtm]: got " << h_traj[0] << ", expected " << expected_day0 << "\n";
            failures++;
        }

        float v_before = h_v[0];
        for (int i = 0; i < p.num_paths; ++i) {
            h_deltas[(p.num_steps - 1) * p.num_paths + i] = 0.9f;
        }
        CUDA_CHECK(cudaMemcpy(d_deltas.get(), h_deltas.data(), h_deltas.size() * sizeof(float), cudaMemcpyHostToDevice));
        evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_deltas.get(), d_v.get(), p);
        CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), p.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::fabs(h_v[0] - v_before) > 1e-5f) {
            std::cerr << "FAIL [no_expiry_rebalance]: V_T changed from " << v_before << " to " << h_v[0] << "\n";
            failures++;
        }
    }

    // test 2: tail-warp BPTT gradients (num_paths = 35 -> 35 % 32 == 3), finite differences, and shift invariance
    {
        SimulationParams p{};
        p.num_paths = 35;
        p.num_steps = 6;
        p.S0 = 100.0f;
        p.K = 100.0f;
        p.r = 0.04f;
        p.sigma = 0.20f;
        p.T = 5.0f / (252.0f * 7.0f);
        p.dt = p.T / (p.num_steps - 1);
        p.cost_ratio = 0.001f;
        p.model = MarketModel::GBM;
        p.option_price = bs_call_price(p.S0, p.K, p.r, p.sigma, p.T);

        auto d_paths = cuda_utils::make_device_buffer<float>(p.num_paths * p.num_steps);
        auto d_vols = cuda_utils::make_device_buffer<float>(p.num_paths * p.num_steps);
        auto d_payoffs = cuda_utils::make_device_buffer<float>(p.num_paths);
        auto d_bs_deltas = cuda_utils::make_device_buffer<float>(p.num_paths * p.num_steps);
        auto d_policy_deltas = cuda_utils::make_device_buffer<float>(p.num_paths * p.num_steps);
        auto d_v = cuda_utils::make_device_buffer<float>(p.num_paths);

        generate_market_paths(d_paths.get(), d_vols.get(), d_payoffs.get(), d_bs_deltas.get(), p, 777ULL);

        std::mt19937 rng(123);
        auto rand_vec = [&](size_t n, float scale) {
            std::normal_distribution<float> dist(0.0f, scale);
            std::vector<float> v(n);
            for (auto& x : v) x = dist(rng);
            return v;
        };

        std::vector<float> h_W1 = rand_vec(hidden_dim * input_dim, 0.2f);
        std::vector<float> h_b1 = rand_vec(hidden_dim, 0.05f);
        std::vector<float> h_W2 = rand_vec(hidden_dim * hidden_dim, 0.1f);
        std::vector<float> h_b2 = rand_vec(hidden_dim, 0.05f);
        std::vector<float> h_W3 = rand_vec(hidden_dim * output_dim, 0.1f);
        std::vector<float> h_b3 = rand_vec(output_dim, 0.05f);

        auto d_W1 = cuda_utils::make_device_buffer<float>(h_W1.size());
        auto d_b1 = cuda_utils::make_device_buffer<float>(h_b1.size());
        auto d_W2 = cuda_utils::make_device_buffer<float>(h_W2.size());
        auto d_b2 = cuda_utils::make_device_buffer<float>(h_b2.size());
        auto d_W3 = cuda_utils::make_device_buffer<float>(h_W3.size());
        auto d_b3 = cuda_utils::make_device_buffer<float>(h_b3.size());

        CUDA_CHECK(cudaMemcpy(d_W1.get(), h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1.get(), h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2.get(), h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2.get(), h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W3.get(), h_W3.data(), h_W3.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3.get(), h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

        auto d_gW1 = cuda_utils::make_device_buffer<float>(h_W1.size());
        auto d_gb1 = cuda_utils::make_device_buffer<float>(h_b1.size());
        auto d_gW2 = cuda_utils::make_device_buffer<float>(h_W2.size());
        auto d_gb2 = cuda_utils::make_device_buffer<float>(h_b2.size());
        auto d_gW3 = cuda_utils::make_device_buffer<float>(h_W3.size());
        auto d_gb3 = cuda_utils::make_device_buffer<float>(h_b3.size());

        auto zero_grads = [&]() {
            CUDA_CHECK(cudaMemset(d_gW1.get(), 0, h_W1.size() * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_gb1.get(), 0, h_b1.size() * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_gW2.get(), 0, h_W2.size() * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_gb2.get(), 0, h_b2.size() * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_gW3.get(), 0, h_W3.size() * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_gb3.get(), 0, h_b3.size() * sizeof(float)));
        };

        MLPWeights weights{d_W1.get(), d_b1.get(), d_W2.get(), d_b2.get(), d_W3.get(), d_b3.get()};
        MLPGradients grads{d_gW1.get(), d_gb1.get(), d_gW2.get(), d_gb2.get(), d_gW3.get(), d_gb3.get()};

        float cvar_alpha = 0.90f;
        float cvar_eps = 0.02f;
        float lambda_cvar = 1.0f;
        bool use_hybrid = true;
        float risk_aversion = 4.0f;

        for (HedgingLossType loss_type : {
                 HedgingLossType::CenteredVarCvar,
                 HedgingLossType::EntropicRisk,
                 HedgingLossType::MeanVarCvar
             }) {
            const char* tag = (loss_type == HedgingLossType::EntropicRisk)
                                  ? "entropic"
                                  : ((loss_type == HedgingLossType::MeanVarCvar) ? "meanvar" : "centered");

            zero_grads();
            forward_pass(d_paths.get(), d_vols.get(), d_policy_deltas.get(), weights, p);
            evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), p);
            std::vector<float> h_v(p.num_paths);
            CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), p.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
            HedgingLossStats stats = compute_hedging_loss_stats(
                h_v, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
            );

            bptt_backward_pass(
                d_paths.get(), d_vols.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(),
                weights, grads, p, stats, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid,
                loss_type, risk_aversion
            );

            auto labeled = [&](const char* param) { return std::string(tag) + ":" + param; };
            failures += check_finite_diff_grad(d_W1.get(), 0, d_gW1.get(), d_paths.get(), d_vols.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, labeled("W1").c_str());
            failures += check_finite_diff_grad(d_W1.get(), 1, d_gW1.get(), d_paths.get(), d_vols.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, labeled("W1").c_str());
            failures += check_finite_diff_grad(d_b1.get(), 1, d_gb1.get(), d_paths.get(), d_vols.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, labeled("b1").c_str());
            failures += check_finite_diff_grad(d_W2.get(), 5, d_gW2.get(), d_paths.get(), d_vols.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, labeled("W2").c_str());
            failures += check_finite_diff_grad(d_b2.get(), 2, d_gb2.get(), d_paths.get(), d_vols.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, labeled("b2").c_str());
            failures += check_finite_diff_grad(d_W3.get(), 7, d_gW3.get(), d_paths.get(), d_vols.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, labeled("W3").c_str());
            failures += check_finite_diff_grad(d_b3.get(), 0, d_gb3.get(), d_paths.get(), d_vols.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, labeled("b3").c_str());

            float gb3_base = 0.0f;
            CUDA_CHECK(cudaMemcpy(&gb3_base, d_gb3.get(), sizeof(float), cudaMemcpyDeviceToHost));

            SimulationParams p_shifted = p;
            float shift = 10.0f;
            p_shifted.option_price += shift;
            zero_grads();
            forward_pass(d_paths.get(), d_vols.get(), d_policy_deltas.get(), weights, p_shifted);
            evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), p_shifted);
            CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), p_shifted.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
            HedgingLossStats shifted_stats = compute_hedging_loss_stats(
                h_v, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
            );

            bptt_backward_pass(
                d_paths.get(), d_vols.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(),
                weights, grads, p_shifted, shifted_stats, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid,
                loss_type, risk_aversion
            );

            float gb3_shifted = 0.0f;
            CUDA_CHECK(cudaMemcpy(&gb3_shifted, d_gb3.get(), sizeof(float), cudaMemcpyDeviceToHost));

            double expected_loss = stats.total_loss;
            if (loss_type == HedgingLossType::EntropicRisk) {
                expected_loss = stats.total_loss - static_cast<double>(shift) * std::exp(p.r * p.T);
            } else if (loss_type == HedgingLossType::MeanVarCvar) {
                expected_loss = stats.total_loss - static_cast<double>(lambda_cvar) * static_cast<double>(shift) * std::exp(p.r * p.T);
            }
            if (std::fabs(expected_loss - shifted_stats.total_loss) > 1e-4) {
                std::cerr << "FAIL [shift_invariance loss " << tag << "]: expected " << expected_loss
                          << " vs " << shifted_stats.total_loss << "\n";
                failures++;
            }
            if (std::fabs(gb3_base - gb3_shifted) > 1e-4f) {
                std::cerr << "FAIL [shift_invariance grad " << tag << "]: " << gb3_base << " vs " << gb3_shifted << "\n";
                failures++;
            }
        }
    }

    // test 3: GBM discounted-price martingale and analytical call price check (65,536 paths)
    {
        SimulationParams p{};
        p.num_paths = 65536;
        p.num_steps = 30;
        p.S0 = 100.0f;
        p.K = 100.0f;
        p.r = 0.05f;
        p.sigma = 0.20f;
        p.T = 30.0f / 252.0f;
        p.dt = p.T / (p.num_steps - 1);
        p.model = MarketModel::GBM;

        size_t total = static_cast<size_t>(p.num_paths) * static_cast<size_t>(p.num_steps);
        auto d_paths = cuda_utils::make_device_buffer<float>(total);
        auto d_vols = cuda_utils::make_device_buffer<float>(total);
        auto d_payoffs = cuda_utils::make_device_buffer<float>(p.num_paths);

        generate_market_paths(d_paths.get(), d_vols.get(), d_payoffs.get(), nullptr, p, 2025ULL);

        std::vector<float> h_ST(p.num_paths);
        std::vector<float> h_payoffs(p.num_paths);
        CUDA_CHECK(cudaMemcpy(h_ST.data(), d_paths.get() + static_cast<size_t>(p.num_steps - 1) * p.num_paths,
                              p.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_payoffs.data(), d_payoffs.get(), p.num_paths * sizeof(float), cudaMemcpyDeviceToHost));

        double disc = std::exp(-static_cast<double>(p.r) * static_cast<double>(p.T));
        double sum_s = 0.0, sum_s2 = 0.0, sum_p = 0.0, sum_p2 = 0.0;
        for (int i = 0; i < p.num_paths; ++i) {
            double ds = disc * static_cast<double>(h_ST[i]);
            double dp = disc * static_cast<double>(h_payoffs[i]);
            sum_s += ds; sum_s2 += ds * ds;
            sum_p += dp; sum_p2 += dp * dp;
        }
        double mean_s = sum_s / p.num_paths;
        double se_s = std::sqrt((sum_s2 / p.num_paths - mean_s * mean_s) / p.num_paths);
        if (std::fabs(mean_s - p.S0) > 3.0 * se_s) {
            std::cerr << "FAIL [gbm_martingale]: disc E[S_T]=" << mean_s << " vs S0=" << p.S0 << " (3*SE=" << 3.0 * se_s << ")\n";
            failures++;
        }

        double mean_p = sum_p / p.num_paths;
        double se_p = std::sqrt((sum_p2 / p.num_paths - mean_p * mean_p) / p.num_paths);
        double bs_c0 = bs_call_price(p.S0, p.K, p.r, p.sigma, p.T);
        if (std::fabs(mean_p - bs_c0) > 3.0 * se_p) {
            std::cerr << "FAIL [gbm_call_price]: MC=" << mean_p << " vs BS=" << bs_c0 << " (3*SE=" << 3.0 * se_p << ")\n";
            failures++;
        }
    }

    // test 4: Heston martingale, non-negative variance, and negative return-variance correlation (65,536 paths)
    {
        SimulationParams p{};
        p.num_paths = 65536;
        p.num_steps = 30;
        p.S0 = 100.0f;
        p.K = 100.0f;
        p.r = 0.05f;
        p.sigma = 0.20f;
        p.T = 30.0f / 252.0f;
        p.dt = p.T / (p.num_steps - 1);
        p.model = MarketModel::Heston;
        p.v0 = 0.04f;
        p.theta = 0.04f;
        p.kappa = 2.0f;
        p.xi = 0.50f;
        p.rho = -0.70f;
        p.heston_substeps = 8;

        size_t total = static_cast<size_t>(p.num_paths) * static_cast<size_t>(p.num_steps);
        auto d_paths = cuda_utils::make_device_buffer<float>(total);
        auto d_vols = cuda_utils::make_device_buffer<float>(total);
        auto d_payoffs = cuda_utils::make_device_buffer<float>(p.num_paths);

        generate_market_paths(d_paths.get(), d_vols.get(), d_payoffs.get(), nullptr, p, 3035ULL);

        std::vector<float> h_paths(total);
        std::vector<float> h_vols(total);
        CUDA_CHECK(cudaMemcpy(h_paths.data(), d_paths.get(), total * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vols.data(), d_vols.get(), total * sizeof(float), cudaMemcpyDeviceToHost));

        double disc = std::exp(-static_cast<double>(p.r) * static_cast<double>(p.T));
        double sum_s = 0.0, sum_s2 = 0.0;
        for (int i = 0; i < p.num_paths; ++i) {
            double ds = disc * static_cast<double>(h_paths[static_cast<size_t>(p.num_steps - 1) * p.num_paths + i]);
            sum_s += ds; sum_s2 += ds * ds;
        }
        double mean_s = sum_s / p.num_paths;
        double se_s = std::sqrt((sum_s2 / p.num_paths - mean_s * mean_s) / p.num_paths);
        if (std::fabs(mean_s - p.S0) > 3.0 * se_s) {
            std::cerr << "FAIL [heston_martingale]: disc E[S_T]=" << mean_s << " vs S0=" << p.S0 << " (3*SE=" << 3.0 * se_s << ")\n";
            failures++;
        }

        // check non-negative vols and negative correlation between log-returns and dv
        double sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_y2 = 0.0, sum_xy = 0.0;
        size_t cnt = 0;
        for (size_t idx = 0; idx < total; ++idx) {
            if (h_vols[idx] < 0.0f || std::isnan(h_vols[idx])) {
                std::cerr << "FAIL [heston_nonneg_vol]: invalid vol=" << h_vols[idx] << "\n";
                failures++;
                break;
            }
        }
        for (int i = 0; i < 4096; ++i) {
            for (int t = 0; t < p.num_steps - 1; ++t) {
                size_t c = static_cast<size_t>(t) * p.num_paths + i;
                size_t n = static_cast<size_t>(t + 1) * p.num_paths + i;
                double r_t = std::log(static_cast<double>(h_paths[n]) / static_cast<double>(h_paths[c]));
                double dv = static_cast<double>(h_vols[n] * h_vols[n]) - static_cast<double>(h_vols[c] * h_vols[c]);
                sum_x += r_t; sum_y += dv;
                sum_x2 += r_t * r_t; sum_y2 += dv * dv;
                sum_xy += r_t * dv;
                cnt++;
            }
        }
        double mx = sum_x / cnt, my = sum_y / cnt;
        double corr = (sum_xy / cnt - mx * my) / std::sqrt((sum_x2 / cnt - mx * mx) * (sum_y2 / cnt - my * my));
        if (corr >= -0.4) {
            std::cerr << "FAIL [heston_corr]: expected strong negative corr for rho=-0.70, got " << corr << "\n";
            failures++;
        }
    }

    // test 5: decoupled AdamW weight decay on weights only (biases untouched when grads = 0)
    {
        auto d_W1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_b1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_W2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_b2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_W3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_b3 = cuda_utils::make_device_buffer<float>(output_dim);

        std::vector<float> init_w1(hidden_dim * input_dim, 1.0f);
        std::vector<float> init_b1(hidden_dim, 1.0f);
        std::vector<float> init_w2(hidden_dim * hidden_dim, 1.0f);
        std::vector<float> init_b2(hidden_dim, 1.0f);
        std::vector<float> init_w3(hidden_dim * output_dim, 1.0f);
        std::vector<float> init_b3(output_dim, 1.0f);

        CUDA_CHECK(cudaMemcpy(d_W1.get(), init_w1.data(), init_w1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1.get(), init_b1.data(), init_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2.get(), init_w2.data(), init_w2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2.get(), init_b2.data(), init_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W3.get(), init_w3.data(), init_w3.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3.get(), init_b3.data(), init_b3.size() * sizeof(float), cudaMemcpyHostToDevice));

        auto d_gW1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_gb1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_gW2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_gb2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_gW3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_gb3 = cuda_utils::make_device_buffer<float>(output_dim);
        CUDA_CHECK(cudaMemset(d_gW1.get(), 0, hidden_dim * input_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gb1.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gW2.get(), 0, hidden_dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gb2.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gW3.get(), 0, hidden_dim * output_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gb3.get(), 0, output_dim * sizeof(float)));

        auto d_mW1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_mb1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_mW2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_mb2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_mW3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_mb3 = cuda_utils::make_device_buffer<float>(output_dim);
        CUDA_CHECK(cudaMemset(d_mW1.get(), 0, hidden_dim * input_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mb1.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mW2.get(), 0, hidden_dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mb2.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mW3.get(), 0, hidden_dim * output_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mb3.get(), 0, output_dim * sizeof(float)));

        auto d_vW1 = cuda_utils::make_device_buffer<float>(hidden_dim * input_dim);
        auto d_vb1 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_vW2 = cuda_utils::make_device_buffer<float>(hidden_dim * hidden_dim);
        auto d_vb2 = cuda_utils::make_device_buffer<float>(hidden_dim);
        auto d_vW3 = cuda_utils::make_device_buffer<float>(hidden_dim * output_dim);
        auto d_vb3 = cuda_utils::make_device_buffer<float>(output_dim);
        CUDA_CHECK(cudaMemset(d_vW1.get(), 0, hidden_dim * input_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vb1.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vW2.get(), 0, hidden_dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vb2.get(), 0, hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vW3.get(), 0, hidden_dim * output_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vb3.get(), 0, output_dim * sizeof(float)));

        MLPWeights weights{d_W1.get(), d_b1.get(), d_W2.get(), d_b2.get(), d_W3.get(), d_b3.get()};
        MLPGradients grads{d_gW1.get(), d_gb1.get(), d_gW2.get(), d_gb2.get(), d_gW3.get(), d_gb3.get()};
        AdamState state{
            d_mW1.get(), d_mb1.get(), d_mW2.get(), d_mb2.get(), d_mW3.get(), d_mb3.get(),
            d_vW1.get(), d_vb1.get(), d_vW2.get(), d_vb2.get(), d_vW3.get(), d_vb3.get()
        };
        AdamParams ap;
        ap.lr = 0.01f;
        ap.weight_decay = 0.1f;
        ap.time_step = 1;

        init_optimizer_weights(weights);
        adam_step(weights, grads, state, ap);

        float w1_0 = 0.0f, b1_0 = 0.0f;
        CUDA_CHECK(cudaMemcpy(&w1_0, d_W1.get(), sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&b1_0, d_b1.get(), sizeof(float), cudaMemcpyDeviceToHost));

        float expected_w = 1.0f - ap.lr * ap.weight_decay * 1.0f;
        if (std::fabs(w1_0 - expected_w) > 1e-6f) {
            std::cerr << "FAIL [adamw_decay]: W1[0]=" << w1_0 << ", expected " << expected_w << "\n";
            failures++;
        }
        if (std::fabs(b1_0 - 1.0f) > 1e-6f) {
            std::cerr << "FAIL [adamw_bias_untouched]: b1[0]=" << b1_0 << ", expected 1.0\n";
            failures++;
        }
    }

    if (failures > 0) {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
