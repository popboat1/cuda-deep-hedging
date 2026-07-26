#ifndef MLP_H
#define MLP_H

#include "simulation.h"

constexpr int input_dim = 3;
constexpr int hidden_dim = 32;
constexpr int output_dim = 1;

struct MLPWeights{
    float* d_W1; // size 32 x 3
    float* d_b1; // size 32
    float* d_W2; // size 32 x 32
    float* d_b2; // size 32
    float* d_W3; // size 1 x 32
    float* d_b3; // size 1
};

// host interface
void forward_pass(
    float* d_paths,
    float* d_policy_deltas,
    const MLPWeights weights,
    const SimulationParams& params
);

#endif