#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "mlp.h"
#include "autograd.h"

struct AdamParams {
    float lr = 0.001f;        // learning rate
    float beta1 = 0.9f;       // exponential decay rate for 1st moment
    float beta2 = 0.999f;     // exponential decay rate for 2nd moment
    float epsilon = 1e-8f;    // numerical stability constant
    int time_step = 1;        // Adam iteration counter (t >= 1)
};

struct AdamState {
    // 1st moment vectors (m)
    float* d_mW1; float* d_mb1;
    float* d_mW2; float* d_mb2;
    float* d_mW3; float* d_mb3;

    // 2nd moment vectors (v)
    float* d_vW1; float* d_vb1;
    float* d_vW2; float* d_vb2;
    float* d_vW3; float* d_vb3;
};

// static device weight pointer initialization
void init_optimizer_weights(const MLPWeights& weights);

// host wrapper
void adam_step(
    MLPWeights& weights,
    MLPGradients& grads,
    AdamState& state,
    AdamParams& params
);

#endif