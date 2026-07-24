#include <iostream>
#include "simulation.h"

int main() {
    try {
        SimulationParams params;
        params.num_paths = 10000;
        params.num_steps = 30;
        params.S0 = 100.0f;
        params.r = 0.05f;
        params.sigma = 0.2f;
        params.T = 1.0f / 12.0f; // 1 month
        params.dt = params.T / params.num_steps;

        std::cout << "allocating vram for " << params.num_paths << " paths..." << std::endl;
        
        // allocate buffer
        size_t total_elements = params.num_paths * params.num_steps;
        auto d_paths = cuda_utils::make_device_buffer<float>(total_elements);

        std::cout << "launching generator..." << std::endl;
        generate_gbm_paths(d_paths.get(), params);

        std::cout << "done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}