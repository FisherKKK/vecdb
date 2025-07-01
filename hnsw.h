#ifndef HNSW_H
#define HNSW_H

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

class HNSW {
public:
    HNSW(int dim, int m, int ef_construction);
    void addPoint(const std::vector<float>& point);
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k);

private:
    struct Node {
        std::vector<float> point;
        std::vector<std::vector<int>> connections;
    };

    int dim_;
    int m_;
    int ef_construction_;
    int max_level_;
    int entry_point_;
    std::vector<Node> nodes_;
    std::mt19937 rng_;

    float distance(const std::vector<float>& a, const std::vector<float>& b);
    int getRandomLevel();
    std::vector<std::pair<int, float>> searchLayer(const std::vector<float>& query, int entry_point, int ef, int level);
};

#endif // HNSW_H
