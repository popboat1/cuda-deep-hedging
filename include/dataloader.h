#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>
#include "simulation.cuh"

struct RealDataset {
    int num_paths = 0;
    int num_steps = 0;
    std::vector<float> h_paths;   // flattened (num_steps * num_paths)
    std::vector<float> h_payoffs; // length num_paths
    std::vector<float> h_deltas;  // benchmark Black-Scholes deltas
    std::vector<float> h_S0_orig; // original unnormalized spot prices
};

// reads real market CSV paths and normalizes initial stock price S0 to params.S0
RealDataset load_and_normalize_real_csv(const std::string& filepath, const SimulationParams& params);

// estimate annualized volatility strictly from dataset 1-step log returns
float estimate_annualized_vol(const RealDataset& ds, float dt);

// compute analytical Black-Scholes benchmark deltas for a loaded dataset
void compute_bs_deltas_host(RealDataset& ds, const SimulationParams& params);

#endif
