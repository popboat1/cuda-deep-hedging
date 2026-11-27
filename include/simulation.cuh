#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include "cuda_utils.cuh"

enum class MarketModel {
    GBM = 0,
    Heston = 1
};

// parameters for GPU path generation and option pricing
struct SimulationParams {
    int num_paths = 0;
    int num_steps = 30;
    float S0 = 100.0f;
    float K = 100.0f;
    float r = 0.05f;
    float sigma = 0.20f;
    float T = 30.0f / 252.0f;
    float dt = (30.0f / 252.0f) / 29.0f;
    float cost_ratio = 0.0010f;
    float option_price = 0.0f;
    MarketModel model = MarketModel::GBM;

    // heston stochastic variance parameters
    float v0 = 0.04f;
    float theta = 0.04f;
    float kappa = 2.0f;
    float xi = 0.50f;
    float rho = -0.70f;
    int heston_substeps = 8;
};

// generate simulated paths, instantaneous volatilities, payoffs, and benchmark BS deltas
void generate_market_paths(
    float* d_paths,
    float* d_vols,
    float* d_payoffs,
    float* d_deltas,
    const SimulationParams& params,
    unsigned long long seed = 1234ULL
);

// estimate risk-neutral discounted call price via independent Monte Carlo draw
float estimate_mc_option_price(
    const SimulationParams& params,
    int calibration_paths = 200000,
    unsigned long long seed = 987654321ULL
);

#endif
