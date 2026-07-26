#ifndef PORTFOLIO_H
#define PORTFOLIO_H

#include "simulation.h"

void evaluate_portfolio(
    const float* d_paths,
    const float* d_payoffs,
    const float* d_deltas,
    float* d_portfolio_values,
    const SimulationParams& params
);

#endif