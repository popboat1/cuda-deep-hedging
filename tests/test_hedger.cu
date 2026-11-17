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

static float bs_call_price(float S, float K, float r, float sigma, float T) {
    float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * std::sqrt(T));
    float d2 = d1 - sigma * std::sqrt(T);
    auto ncdf = [](float x) { return 0.5f * std::erfc(-x * static_cast<float>(M_SQRT1_2)); };
    return S * ncdf(d1) - K * std::exp(-r * T) * ncdf(d2);
}

static double evaluate_loss_for_weights(
    const float* d_paths,
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
    forward_pass(const_cast<float*>(d_paths), d_policy_deltas, weights, p);
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

    float eps = 1e-2f;
    float plus_val = orig_val + eps;
    float minus_val = orig_val - eps;

    CUDA_CHECK(cudaMemcpy(d_weight_buf + param_idx, &plus_val, sizeof(float), cudaMemcpyHostToDevice));
    double loss_plus = evaluate_loss_for_weights(
        d_paths, d_payoffs, d_policy_deltas, d_v, weights, p,
        cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
    );

    CUDA_CHECK(cudaMemcpy(d_weight_buf + param_idx, &minus_val, sizeof(float), cudaMemcpyHostToDevice));
    double loss_minus = evaluate_loss_for_weights(
        d_paths, d_payoffs, d_policy_deltas, d_v, weights, p,
        cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
    );

    CUDA_CHECK(cudaMemcpy(d_weight_buf + param_idx, &orig_val, sizeof(float), cudaMemcpyHostToDevice));

    double fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double abs_err = std::fabs(fd_grad - static_cast<double>(anal_grad));
    double denom = std::max(1e-2, std::max(std::fabs(fd_grad), std::fabs(static_cast<double>(anal_grad))));
    double rel_err = abs_err / denom;

    if (rel_err > 0.03 && abs_err > 5e-4) {
        std::cerr << "FAIL [fd_grad " << name << "]: anal=" << anal_grad
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

        // consistency with non-zero delta and analytical BS option_price
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

        // expiry rebalance invariance: changing delta at t = num_steps - 1 must not change V_T
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
        p.option_price = bs_call_price(p.S0, p.K, p.r, p.sigma, p.T);

        auto d_paths = cuda_utils::make_device_buffer<float>(p.num_paths * p.num_steps);
        auto d_payoffs = cuda_utils::make_device_buffer<float>(p.num_paths);
        auto d_bs_deltas = cuda_utils::make_device_buffer<float>(p.num_paths * p.num_steps);
        auto d_policy_deltas = cuda_utils::make_device_buffer<float>(p.num_paths * p.num_steps);
        auto d_v = cuda_utils::make_device_buffer<float>(p.num_paths);

        generate_gbm_paths(d_paths.get(), d_payoffs.get(), d_bs_deltas.get(), p, 777ULL);

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

        for (HedgingLossType loss_type : {HedgingLossType::CenteredVarCvar, HedgingLossType::EntropicRisk}) {
            const char* tag = (loss_type == HedgingLossType::EntropicRisk) ? "entropic" : "centered";

            zero_grads();
            forward_pass(d_paths.get(), d_policy_deltas.get(), weights, p);
            evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), p);
            std::vector<float> h_v(p.num_paths);
            CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), p.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
            HedgingLossStats stats = compute_hedging_loss_stats(
                h_v, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
            );

            bptt_backward_pass(
                d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(),
                weights, grads, p, stats, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid,
                loss_type, risk_aversion
            );

            failures += check_finite_diff_grad(d_W1.get(), 0, d_gW1.get(), d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, tag);
            failures += check_finite_diff_grad(d_W1.get(), 1, d_gW1.get(), d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, tag);
            failures += check_finite_diff_grad(d_b1.get(), 0, d_gb1.get(), d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, tag);
            failures += check_finite_diff_grad(d_W2.get(), 5, d_gW2.get(), d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, tag);
            failures += check_finite_diff_grad(d_b2.get(), 3, d_gb2.get(), d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, tag);
            failures += check_finite_diff_grad(d_W3.get(), 7, d_gW3.get(), d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, tag);
            failures += check_finite_diff_grad(d_b3.get(), 0, d_gb3.get(), d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), weights, p, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion, tag);

            float gb3_base = 0.0f;
            CUDA_CHECK(cudaMemcpy(&gb3_base, d_gb3.get(), sizeof(float), cudaMemcpyDeviceToHost));

            // shift option_price by +10 and verify cash/shift invariance
            SimulationParams p_shifted = p;
            float shift = 10.0f;
            p_shifted.option_price += shift;
            zero_grads();
            forward_pass(d_paths.get(), d_policy_deltas.get(), weights, p_shifted);
            evaluate_portfolio(d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(), p_shifted);
            CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), p_shifted.num_paths * sizeof(float), cudaMemcpyDeviceToHost));
            HedgingLossStats shifted_stats = compute_hedging_loss_stats(
                h_v, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid, loss_type, risk_aversion
            );

            bptt_backward_pass(
                d_paths.get(), d_payoffs.get(), d_policy_deltas.get(), d_v.get(),
                weights, grads, p_shifted, shifted_stats, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid,
                loss_type, risk_aversion
            );

            float gb3_shifted = 0.0f;
            CUDA_CHECK(cudaMemcpy(&gb3_shifted, d_gb3.get(), sizeof(float), cudaMemcpyDeviceToHost));

            double expected_loss = (loss_type == HedgingLossType::EntropicRisk)
                                   ? (stats.total_loss - static_cast<double>(shift) * std::exp(p.r * p.T))
                                   : stats.total_loss;
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

    if (failures > 0) {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
