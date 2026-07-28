#ifndef AUTOGRAD_CUH
#define AUTOGRAD_CUH

#include "mlp.cuh"
#include "simulation.cuh"

struct MLPGradients {
    float* d_gW1; // size 32 x 3
    float* d_gb1; // size 32
    float* d_gW2; // size 32 x 32
    float* d_gb2; // size 32
    float* d_gW3; // size 1 x 32
    float* d_gb3; // size 1
};

// host wrapper
void bptt_backward_pass(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_policy_deltas,
    const float* d_portfolio_values,
    const MLPWeights& weights,
    MLPGradients& grads,
    const SimulationParams& params,
    float var_cutoff = 0.0f,
    float cvar_alpha = 0.95f,
    float lambda_cvar = 1.0f,
    bool use_hybrid = true
);

#endif