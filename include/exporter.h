#ifndef EXPORTER_H
#define EXPORTER_H

#include <string>
#include "mlp.h"
#include "simulation.h"

void export_hedging_surface(
    const std::string& filename,
    const MLPWeights& weights,
    const SimulationParams& params,
    float prev_delta_fixed = 0.5f,
    int num_s_points = 100,
    int num_tau_points = 100
);

#endif