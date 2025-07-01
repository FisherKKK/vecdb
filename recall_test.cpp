#include "hnsw.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

// Function to generate random data
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

// Function for brute-force k-NN search
std::vector<int> brute_force_knn(const std::vector<std::vector<float>>& data, const std::vector<float>& query, int k) {
    std::vector<std::pair<float, int>> distances;
    for (int i = 0; i < data.size(); ++i) {
        float dist = 0.0f;
        for (int j = 0; j < query.size(); ++j) {
            dist += (data[i][j] - query[j]) * (data[i][j] - query[j]);
        }
        distances.push_back({std::sqrt(dist), i});
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> result;
    for (int i = 0; i < k && i < distances.size(); ++i) {
        result.push_back(distances[i].second);
    }
    return result;
}

int main() {
    int num_vectors = 1000;
    int dim = 16;
    int m = 16;
    int ef_construction = 200;
    int k = 10;
    int num_queries = 100;

    // Generate data and queries
    auto data = generate_random_data(num_vectors, dim);
    auto queries = generate_random_data(num_queries, dim);

    // Build HNSW index
    HNSW hnsw(dim, m, ef_construction);
    for (const auto& p : data) {
        hnsw.addPoint(p);
    }

    // Test recall
    double total_recall = 0.0;
    for (const auto& query : queries) {
        // Ground truth
        auto true_neighbors = brute_force_knn(data, query, k);

        // HNSW search
        auto hnsw_results = hnsw.search(query, k);

        // Calculate recall for this query
        std::set<int> true_set(true_neighbors.begin(), true_neighbors.end());
        int found_count = 0;
        for (const auto& result : hnsw_results) {
            if (true_set.count(result.first)) {
                found_count++;
            }
        }
        total_recall += (double)found_count / k;
    }

    double average_recall = total_recall / num_queries;
    std::cout << "HNSW Recall@" << k << ": " << average_recall << std::endl;

    return 0;
}
