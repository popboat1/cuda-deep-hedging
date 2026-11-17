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

    if (ds.num_paths <= 0 || ds.num_steps <= 1) {
        throw std::runtime_error("Invalid dataset header in CSV: " + filepath);
    }

    ds.h_paths.resize(ds.num_paths * ds.num_steps);
    ds.h_payoffs.resize(ds.num_paths);
    ds.h_deltas.resize(ds.num_paths * ds.num_steps);
    ds.h_S0_orig.resize(ds.num_paths);

    int row = 0;
    while (std::getline(file, line) && row < ds.num_paths) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;

        // skip original strike & payoff
        std::getline(ss, token, ',');
        std::getline(ss, token, ',');

        // read price series for num_steps steps
        std::vector<float> raw_prices;
        raw_prices.reserve(ds.num_steps);
        while (std::getline(ss, token, ',')) {
            raw_prices.push_back(std::stof(token));
        }

        if (static_cast<int>(raw_prices.size()) != ds.num_steps) {
            throw std::runtime_error("Row step count mismatch in CSV: " + filepath);
        }

        float S0_orig = raw_prices[0];
        ds.h_S0_orig[row] = S0_orig;
        float scale = params.S0 / S0_orig; // normalization scale factor

        // normalized terminal payoff
        float S_T_norm = raw_prices.back() * scale;
        ds.h_payoffs[row] = (S_T_norm > params.K) ? (S_T_norm - params.K) : 0.0f;

        // store normalized price path in [t, path_idx] layout
        for (int t = 0; t < ds.num_steps; ++t) {
            ds.h_paths[t * ds.num_paths + row] = raw_prices[t] * scale;
        }
        row++;
    }

    file.close();
    if (row != ds.num_paths) {
        throw std::runtime_error("Dataset row count mismatch in CSV: " + filepath);
    }

    if (params.sigma > 0.0f) {
        compute_bs_deltas_host(ds, params);
    }

    std::cout << "loaded " << row << " real paths from " << filepath << std::endl;
    return ds;
}

float estimate_annualized_vol(const RealDataset& ds, float dt) {
    if (ds.num_paths <= 0 || ds.num_steps <= 1 || dt <= 0.0f) {
        throw std::invalid_argument("invalid dataset or dt for volatility estimation");
    }

    double sum_r = 0.0;
    double sum_r2 = 0.0;
    size_t count = 0;

    for (int t = 0; t < ds.num_steps - 1; ++t) {
        for (int i = 0; i < ds.num_paths; ++i) {
            float s_curr = ds.h_paths[t * ds.num_paths + i];
            float s_next = ds.h_paths[(t + 1) * ds.num_paths + i];
            double ret = std::log(static_cast<double>(s_next) / static_cast<double>(s_curr));
            sum_r += ret;
            sum_r2 += ret * ret;
            count++;
        }
    }

    if (count < 2) {
        throw std::runtime_error("insufficient transitions to estimate volatility");
    }

    double mean_r = sum_r / static_cast<double>(count);
    double var_r = (sum_r2 - static_cast<double>(count) * mean_r * mean_r) / static_cast<double>(count - 1);
    double vol = std::sqrt(std::max(0.0, var_r) / static_cast<double>(dt));
    return static_cast<float>(vol);
}

void compute_bs_deltas_host(RealDataset& ds, const SimulationParams& params) {
    float sigma_sq = params.sigma * params.sigma;
    float num_term = params.r + 0.5f * sigma_sq;

    for (int row = 0; row < ds.num_paths; ++row) {
        for (int t = 0; t < ds.num_steps; ++t) {
            float S_t = ds.h_paths[t * ds.num_paths + row];
            float tau = params.T - (t * params.dt);
            if (tau <= 0.0f || t == ds.num_steps - 1) {
                ds.h_deltas[t * ds.num_paths + row] = (S_t > params.K) ? 1.0f : 0.0f;
            } else {
                float d1 = (std::log(S_t / params.K) + (num_term * tau)) / (params.sigma * std::sqrt(tau));
                ds.h_deltas[t * ds.num_paths + row] = 0.5f * std::erfc(-d1 * static_cast<float>(M_SQRT1_2));
            }
        }
    }
}
