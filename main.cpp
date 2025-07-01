#include "hnsw.h"
#include "ivf.h"
#include <iostream>
#include <vector>
#include <random>

std::vector<std::vector<float>> generate_random_data(int num_vectors, int dim) {
    std::vector<std::vector<float>> data(num_vectors, std::vector<float>(dim));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<> dist(0, 1);
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < dim; ++j) {
            data[i][j] = dist(rng);
        }
    }
    return data;
}

int main() {
    int num_vectors = 1000;
    int dim = 16;
    int m = 16;
    int ef_construction = 200;
    int k = 5;
    int nlist = 10;
    int nprobe = 2;

    // Generate data
    auto data = generate_random_data(num_vectors, dim);
    auto query = generate_random_data(1, dim)[0];

    // HNSW
    HNSW hnsw(dim, m, ef_construction);
    for (const auto& p : data) {
        hnsw.addPoint(p);
    }
    auto hnsw_results = hnsw.search(query, k);

    std::cout << "HNSW Results:" << std::endl;
    for (const auto& result : hnsw_results) {
        std::cout << "  Index: " << result.first << ", Distance: " << result.second << std::endl;
    }

    // IVF
    IVF ivf(dim, nlist);
    ivf.train(data);
    for (const auto& p : data) {
        ivf.add(p);
    }
    auto ivf_results = ivf.search(query, k, nprobe);

    std::cout << "\nIVF Results:" << std::endl;
    for (const auto& result : ivf_results) {
        std::cout << "  Index: " << result.first << ", Distance: " << result.second << std::endl;
    }

    return 0;
}
