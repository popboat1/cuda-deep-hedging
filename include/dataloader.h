#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>
#include "simulation.cuh"

struct RealDataset {
    int num_paths = 0;
    int num_steps = 0;
    std::vector<float> h_paths;   // flattened (num_paths * num_steps)
    std::vector<float> h_payoffs; // length num_paths
    std::vector<float> h_deltas;  // benchmark Black-Scholes deltas
};

// reads real market CSV paths and normalizes initial stock price S0 to params.S0
RealDataset load_and_normalize_real_csv(const std::string& filepath, const SimulationParams& params);

#endif