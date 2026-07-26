#ifndef SIMULATION_H
#define SIMULATION_H

#include "cuda_utils.h"

// struct for CUDA kernel params
struct SimulationParams {
    int num_paths;     // sim nums e.g., 1,000,000
    int num_steps;     // how long? e.g., 30 days
    float S0;          // initial stock price
    float r;           // risk-free rate
    float sigma;       // volatility
    float T;           // time to expiry (years)
    float dt;          // T / num_steps
    float K;           // strike price e.g., 100.0
    float cost_ratio;  // transaction fee rate e.g., 0.001f (10 bps)
    float option_price;// simulated/black-scholes option price
};

// host interface that wraps around kernel launch in simulation.cu
void generate_gbm_paths( 
    float* d_paths,                    // pre-allocated device buffer [num_paths x num_steps]
    float* d_payoffs,
    float* d_deltas,
    const SimulationParams& params,
    unsigned long long seed = 1234ULL
);

#endif