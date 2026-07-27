#include "dataloader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <stdexcept>

RealDataset load_and_normalize_real_csv(const std::string& filepath, const SimulationParams& params) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset CSV: " + filepath);
    }

    RealDataset ds;
    std::string line;

    // read header line: num_paths, num_steps
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string p_str, s_str;
        if (std::getline(ss, p_str, ',') && std::getline(ss, s_str, ',')) {
            ds.num_paths = std::stoi(p_str);
            ds.num_steps = std::stoi(s_str);
        }
    }

    ds.h_paths.resize(ds.num_paths * ds.num_steps);
    ds.h_payoffs.resize(ds.num_paths);
    ds.h_deltas.resize(ds.num_paths * ds.num_steps);

    float sigma_sq = params.sigma * params.sigma;
    float num_term = params.r + 0.5f * sigma_sq;

    int row = 0;
    while (std::getline(file, line) && row < ds.num_paths) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;

        // skip original strike & payoff
        std::getline(ss, token, ',');
        std::getline(ss, token, ',');

        // read price series for 30 steps
        std::vector<float> raw_prices;
        raw_prices.reserve(ds.num_steps);
        while (std::getline(ss, token, ',')) {
            raw_prices.push_back(std::stof(token));
        }

        float S0_orig = raw_prices[0];
        float scale = params.S0 / S0_orig; // normalization scale factor

        // normalized terminal payoff
        float S_T_norm = raw_prices.back() * scale;
        ds.h_payoffs[row] = (S_T_norm > params.K) ? (S_T_norm - params.K) : 0.0f;

        // normalized price path & Black-Scholes benchmark delta
        int path_offset = row * ds.num_steps;
        for (int t = 0; t < ds.num_steps; ++t) {
            float S_t = raw_prices[t] * scale;
            ds.h_paths[t * ds.num_paths + row] = S_t;

            float tau = params.T - (t * params.dt);
            if (tau <= 0.0f || t == ds.num_steps - 1) {
                ds.h_deltas[t * ds.num_paths + row] = (S_t > params.K) ? 1.0f : 0.0f;
            } else {
                float d1 = (std::log(S_t / params.K) + (num_term * tau)) / (params.sigma * std::sqrt(tau));
                ds.h_deltas[t * ds.num_paths + row] = 0.5f * std::erfc(-d1 * M_SQRT1_2);
            }
        }
        row++;
    }

    file.close();
    std::cout << "loaded " << row << " real paths from " << filepath << std::endl;
    return ds;
}