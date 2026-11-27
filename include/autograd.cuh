#ifndef AUTOGRAD_CUH
#define AUTOGRAD_CUH

#include <vector>
#include "mlp.cuh"
#include "simulation.cuh"

struct MLPGradients {
    float* d_gW1; // size hidden_dim x input_dim
    float* d_gb1; // size hidden_dim
    float* d_gW2; // size hidden_dim x hidden_dim
    float* d_gb2; // size hidden_dim
    float* d_gW3; // size output_dim x hidden_dim
    float* d_gb3; // size output_dim
};

enum class HedgingLossType {
    CenteredVarCvar = 0,
    EntropicRisk = 1,
    MeanVarCvar = 2
};

struct HedgingLossStats {
    double variance = 0.0;        // centered variance E[(V - mean)^2]
    double ru_cvar = 0.0;         // smooth Rockafellar-Uryasev CVaR
    double entropic_risk = 0.0;   // (1 / gamma) * log E[exp(-gamma * V)]
    double total_loss = 0.0;      // active objective value
    float mean_v = 0.0f;          // sample mean E[V]
    float nu_star = 0.0f;         // optimal RU threshold
    float mean_phi_prime = 0.0f;  // mean smooth hinge derivative for centered CVaR
    float log_normalizer = 0.0f;  // log(sum_i exp(-gamma * V_i)) for entropic weights
};

// compute hedging loss statistics and auxiliary scalar multipliers
HedgingLossStats compute_hedging_loss_stats(
    const std::vector<float>& portfolio_values,
    float cvar_alpha = 0.95f,
    float cvar_eps = 0.02f,
    float lambda_cvar = 0.25f,
    bool use_hybrid = true,
    HedgingLossType loss_type = HedgingLossType::MeanVarCvar,
    float risk_aversion = 4.0f,
    bool force_cvar_metric = false
);

// host wrapper
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
    float cvar_alpha = 0.95f,
    float cvar_eps = 0.02f,
    float lambda_cvar = 0.25f,
    bool use_hybrid = true,
    HedgingLossType loss_type = HedgingLossType::MeanVarCvar,
    float risk_aversion = 4.0f
);

#endif
