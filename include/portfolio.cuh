#ifndef PORTFOLIO_CUH
#define PORTFOLIO_CUH

#include "simulation.cuh"

// terminal portfolio evaluation
void evaluate_portfolio(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_deltas,
    float* d_portfolio_values,
    const SimulationParams& params
);

// step-by-step portfolio trajectory evaluation
void evaluate_portfolio_trajectories(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_deltas,
    float* d_trajectories,
    const SimulationParams& params
);

#endif