#include "autograd.cuh"
#include "mlp.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

__device__ inline float signf(float x) {
    if (x > 0.0f) return 1.0f;
    if (x < 0.0f) return -1.0f;
    return 0.0f;
}

// warp reduction using full-mask shuffle instructions
__device__ inline float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// uniform warp reduction into shared memory accumulator
__device__ inline void warp_atomic_add(float* address, float contrib) {
    float sum = warpReduceSum(contrib);
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(address, sum);
    }
}

HedgingLossStats compute_hedging_loss_stats(
    const std::vector<float>& portfolio_values,
    float cvar_alpha,
    float cvar_eps,
    float lambda_cvar,
    bool use_hybrid,
    HedgingLossType loss_type,
    float risk_aversion,
    bool force_cvar_metric
) {
    if (portfolio_values.empty()) {
        throw std::invalid_argument("portfolio_values cannot be empty");
    }
    if (cvar_alpha <= 0.0f || cvar_alpha >= 1.0f) {
        throw std::invalid_argument("cvar_alpha must lie in (0, 1)");
    }
    if (cvar_eps <= 0.0f) {
        throw std::invalid_argument("cvar_eps must be positive");
    }
    if (risk_aversion <= 0.0f) {
        throw std::invalid_argument("risk_aversion must be positive");
    }

    size_t n = portfolio_values.size();
    double inv_n = 1.0 / static_cast<double>(n);

    double sum_v = 0.0;
    for (float v : portfolio_values) {
        sum_v += static_cast<double>(v);
    }
    double mean_v = sum_v * inv_n;

    bool center_cvar_losses = (loss_type != HedgingLossType::MeanVarCvar);
    bool need_cvar = (use_hybrid && lambda_cvar > 0.0f) || force_cvar_metric;
    bool need_entropic = (loss_type == HedgingLossType::EntropicRisk);

    double sum_sq = 0.0;
    std::vector<float> cvar_losses;
    if (need_cvar) cvar_losses.resize(n);
    double max_neg_gamma_v = std::numeric_limits<double>::lowest();
    double gamma = static_cast<double>(risk_aversion);

    for (size_t i = 0; i < n; ++i) {
        double v_i = static_cast<double>(portfolio_values[i]);
        double v_tilde = v_i - mean_v;
        sum_sq += v_tilde * v_tilde;
        if (need_cvar) {
            cvar_losses[i] = static_cast<float>(center_cvar_losses ? (-v_tilde) : (-v_i));
        }
        if (need_entropic) {
            max_neg_gamma_v = std::max(max_neg_gamma_v, -gamma * v_i);
        }
    }

    HedgingLossStats stats;
    stats.variance = sum_sq * inv_n;
    stats.mean_v = static_cast<float>(mean_v);

    if (need_entropic) {
        double sum_exp = 0.0;
        for (float v : portfolio_values) {
            sum_exp += std::exp(-gamma * static_cast<double>(v) - max_neg_gamma_v);
        }
        double log_sum_exp = max_neg_gamma_v + std::log(sum_exp);
        stats.entropic_risk = (log_sum_exp + std::log(inv_n)) / gamma;
        stats.log_normalizer = static_cast<float>(log_sum_exp);
    }

    if (need_cvar) {
        // bracket smooth RU threshold around empirical quantile to avoid wide global bisection
        double eps = static_cast<double>(cvar_eps);
        double inv_eps = 1.0 / eps;
        double half_eps = 0.5 * eps;
        double tail_prob = 1.0 - static_cast<double>(cvar_alpha);

        std::vector<float> q_buf = cvar_losses;
        size_t q_idx = (n > 1) ? std::min(n - 1, static_cast<size_t>(std::floor(static_cast<double>(n - 1) * static_cast<double>(cvar_alpha)))) : 0;
        std::nth_element(q_buf.begin(), q_buf.begin() + q_idx, q_buf.end());
        double q_est = static_cast<double>(q_buf[q_idx]);

        double lo = q_est - eps;
        double hi = q_est + eps;

        auto mean_smooth_exceedance = [&](double nu) {
            double s = 0.0;
            for (float loss_f : cvar_losses) {
                double u = (static_cast<double>(loss_f) - nu + half_eps) * inv_eps;
                if (u >= 1.0) {
                    s += 1.0;
                } else if (u > 0.0) {
                    s += u;
                }
            }
            return s * inv_n;
        };

        // expand bracket only if ties or discrete steps push root outside +/- eps
        while (mean_smooth_exceedance(lo) < tail_prob) lo -= eps;
        while (mean_smooth_exceedance(hi) > tail_prob) hi += eps;

        for (int iter = 0; iter < 32; ++iter) {
            double mid = 0.5 * (lo + hi);
            if (mean_smooth_exceedance(mid) > tail_prob) {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo < 1e-7) break;
        }

        double nu_star = 0.5 * (lo + hi);
        double sum_psi = 0.0;
        double sum_phi_prime = 0.0;

        for (float loss_f : cvar_losses) {
            double z = static_cast<double>(loss_f) - nu_star;
            double u = z + half_eps;
            if (u >= eps) {
                sum_psi += z;
                sum_phi_prime += 1.0;
            } else if (u > 0.0) {
                sum_psi += (u * u) * (0.5 * inv_eps);
                sum_phi_prime += u * inv_eps;
            }
        }

        stats.ru_cvar = nu_star + (sum_psi * inv_n) / tail_prob;
        stats.nu_star = static_cast<float>(nu_star);
        stats.mean_phi_prime = static_cast<float>(sum_phi_prime * inv_n);
    }

    if (loss_type == HedgingLossType::EntropicRisk) {
        stats.total_loss = stats.entropic_risk;
    } else {
        stats.total_loss = stats.variance + ((use_hybrid && lambda_cvar > 0.0f) ? static_cast<double>(lambda_cvar) * stats.ru_cvar : 0.0);
    }

    return stats;
}

__global__ void bptt_backward(
    const float* d_paths,
    const float* d_vols,
    const float* d_payoffs,
    const float* d_policy_deltas,
    const float* d_portfolio_values,
    MLPGradients grads,
    const SimulationParams params,
    const HedgingLossStats loss_stats,
    float cvar_alpha,
    float cvar_eps,
    float lambda_cvar,
    bool use_hybrid,
    int loss_type_int,
    float risk_aversion
){
    __shared__ float s_gW1[hidden_dim * input_dim];
    __shared__ float s_gb1[hidden_dim];
    __shared__ float s_gW2[hidden_dim * hidden_dim];
    __shared__ float s_gb2[hidden_dim];
    __shared__ float s_gW3[hidden_dim * output_dim];
    __shared__ float s_gb3[output_dim];

    int tid = threadIdx.x;

    for (int i = tid; i < hidden_dim * input_dim; i += blockDim.x) s_gW1[i] = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) s_gb1[i] = 0.0f;
    for (int i = tid; i < hidden_dim * hidden_dim; i += blockDim.x) s_gW2[i] = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) s_gb2[i] = 0.0f;
    for (int i = tid; i < hidden_dim * output_dim; i += blockDim.x) s_gW3[i] = 0.0f;
    for (int i = tid; i < output_dim; i += blockDim.x) s_gb3[i] = 0.0f;

    __syncthreads();

    int path_idx = threadIdx.x + blockIdx.x * blockDim.x;
    bool active = (path_idx < params.num_paths);
    bool warp_has_active = __any_sync(0xFFFFFFFF, active);

    float dL_dV_T = 0.0f;
    if (active) {
        float V_T = d_portfolio_values[path_idx];
        if (loss_type_int == static_cast<int>(HedgingLossType::EntropicRisk)) {
            dL_dV_T = -expf(-risk_aversion * V_T - loss_stats.log_normalizer);
        } else {
            float v_tilde = V_T - loss_stats.mean_v;
            float n_paths = static_cast<float>(params.num_paths);

            dL_dV_T = (2.0f / n_paths) * v_tilde;

            if (use_hybrid && lambda_cvar > 0.0f) {
                bool is_mean_var_cvar = (loss_type_int == static_cast<int>(HedgingLossType::MeanVarCvar));
                float loss_i = is_mean_var_cvar ? (-V_T) : (-v_tilde);
                float z = loss_i - loss_stats.nu_star;
                float u = (z + 0.5f * cvar_eps) / cvar_eps;
                float phi_prime = fminf(1.0f, fmaxf(0.0f, u));
                float tail_scale = lambda_cvar / (n_paths * (1.0f - cvar_alpha));

                // uncentered RU-CVaR penalizes transaction fee bleed while remaining invariant to option_price shifts
                float cvar_term = is_mean_var_cvar ? phi_prime : (phi_prime - loss_stats.mean_phi_prime);
                dL_dV_T -= tail_scale * cvar_term;
            }
        }
    }

    float exp_r_dt = expf(params.r * params.dt);
    float accrual_to_T = 1.0f;
    float upstream_d_delta = 0.0f;

    // all lanes iterate uniformly so tail-warp shuffle reductions never diverge
    for (int t = params.num_steps - 2; t >= 0; --t) {
        float x[5] = {};
        float h1[hidden_dim] = {};
        float h2[hidden_dim] = {};
        float d1[hidden_dim] = {};
        float d2[hidden_dim] = {};
        float d3 = 0.0f;

        if (active) {
            int curr_idx = t * params.num_paths + path_idx;
            float S_t = d_paths[curr_idx];
            float delta_t = d_policy_deltas[curr_idx];
            float prev_delta = (t > 0) ? d_policy_deltas[(t - 1) * params.num_paths + path_idx] : 0.0f;

            compute_state_vector(d_paths, d_vols, t, path_idx, prev_delta, params, x);

            for (int i = 0; i < hidden_dim; ++i) {
                float sum = c_b1[i];
                for (int j = 0; j < input_dim; ++j) {
                    sum += c_W1[i * input_dim + j] * x[j];
                }
                h1[i] = fmaxf(0.0f, sum);
            }

            for (int i = 0; i < hidden_dim; ++i) {
                float sum = c_b2[i];
                for (int j = 0; j < hidden_dim; ++j) {
                    sum += c_W2[i * hidden_dim + j] * h1[j];
                }
                h2[i] = fmaxf(0.0f, sum);
            }

            float sign_t = signf(delta_t - prev_delta);
            float dVT_ddelta = 0.0f;

            int next_idx = (t + 1) * params.num_paths + path_idx;
            float S_next = d_paths[next_idx];

            if (t == params.num_steps - 2) {
                float sign_liq = signf(delta_t);
                dVT_ddelta = -S_t * (1.0f + params.cost_ratio * sign_t) * exp_r_dt
                             + S_next * (1.0f - params.cost_ratio * sign_liq);
            } else {
                float next_delta = d_policy_deltas[next_idx];
                float sign_next = signf(next_delta - delta_t);

                float step_return = -S_t * (1.0f + params.cost_ratio * sign_t) * exp_r_dt
                                    + S_next * (1.0f + params.cost_ratio * sign_next);

                dVT_ddelta = step_return * accrual_to_T;
            }

            float dL_ddelta = (dL_dV_T * dVT_ddelta) + upstream_d_delta;
            d3 = dL_ddelta * (delta_t * (1.0f - delta_t));

            for (int i = 0; i < hidden_dim; ++i) {
                float incoming = c_W3[i] * d3;
                d2[i] = (h2[i] > 0.0f) ? incoming : 0.0f;
            }

            for (int i = 0; i < hidden_dim; ++i) {
                float incoming = 0.0f;
                for (int j = 0; j < hidden_dim; ++j) {
                    incoming += c_W2[j * hidden_dim + i] * d2[j];
                }
                d1[i] = (h1[i] > 0.0f) ? incoming : 0.0f;
            }

            upstream_d_delta = 0.0f;
            for (int i = 0; i < hidden_dim; ++i) {
                upstream_d_delta += c_W1[i * input_dim + 1] * d1[i];
            }
            upstream_d_delta *= d_x1_d_prev_delta;
        }

        // all 32 warp lanes enter reduction uniformly with inactive contrib = 0
        if (warp_has_active) {
            warp_atomic_add(&s_gb3[0], active ? d3 : 0.0f);
            for (int j = 0; j < hidden_dim; ++j) {
                warp_atomic_add(&s_gW3[j], active ? (d3 * h2[j]) : 0.0f);
            }

            for (int i = 0; i < hidden_dim; ++i) {
                warp_atomic_add(&s_gb2[i], active ? d2[i] : 0.0f);
                for (int j = 0; j < hidden_dim; ++j) {
                    warp_atomic_add(&s_gW2[i * hidden_dim + j], active ? (d2[i] * h1[j]) : 0.0f);
                }
            }

            for (int i = 0; i < hidden_dim; ++i) {
                warp_atomic_add(&s_gb1[i], active ? d1[i] : 0.0f);
                for (int j = 0; j < input_dim; ++j) {
                    warp_atomic_add(&s_gW1[i * input_dim + j], active ? (d1[i] * x[j]) : 0.0f);
                }
            }
        }

        // step backward one interval so accrual_to_T tracks exp(r * ((N - 2) - t) * dt)
        accrual_to_T *= exp_r_dt;
    }

    __syncthreads();

    for (int i = tid; i < hidden_dim * input_dim; i += blockDim.x) atomicAdd(&grads.d_gW1[i], s_gW1[i]);
    for (int i = tid; i < hidden_dim; i += blockDim.x) atomicAdd(&grads.d_gb1[i], s_gb1[i]);
    for (int i = tid; i < hidden_dim * hidden_dim; i += blockDim.x) atomicAdd(&grads.d_gW2[i], s_gW2[i]);
    for (int i = tid; i < hidden_dim; i += blockDim.x) atomicAdd(&grads.d_gb2[i], s_gb2[i]);
    for (int i = tid; i < hidden_dim * output_dim; i += blockDim.x) atomicAdd(&grads.d_gW3[i], s_gW3[i]);
    for (int i = tid; i < output_dim; i += blockDim.x) atomicAdd(&grads.d_gb3[i], s_gb3[i]);
}

void bptt_backward_pass(
    const float* d_paths,
    const float* d_vols,
    const float* d_payoffs,
    const float* d_policy_deltas,
    const float* d_portfolio_values,
    const MLPWeights& weights,
    MLPGradients& grads,
    const SimulationParams& params,
    const HedgingLossStats& loss_stats,
    float cvar_alpha,
    float cvar_eps,
    float lambda_cvar,
    bool use_hybrid,
    HedgingLossType loss_type,
    float risk_aversion
){
    // weights are already resident in constant memory from forward_pass
    int threads = 256;
    int blocks = cuda_utils::ceil_div(params.num_paths, threads);

    bptt_backward<<<blocks, threads>>>(
        d_paths, d_vols, d_payoffs, d_policy_deltas, d_portfolio_values,
        grads, params, loss_stats, cvar_alpha, cvar_eps, lambda_cvar, use_hybrid,
        static_cast<int>(loss_type), risk_aversion
    );
    CUDA_CHECK_KERNEL();
}
