#ifndef IVF_H
#define IVF_H

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <map>

class IVF {
public:
    IVF(int dim, int nlist);
    void train(const std::vector<std::vector<float>>& data);
    void add(const std::vector<float>& point);
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k, int nprobe);

private:
    int dim_;
    int nlist_;
    std::vector<std::vector<float>> centroids_;
    std::vector<std::vector<int>> inverted_lists_;
    std::vector<std::vector<float>> original_points_;
    int original_index_counter_ = 0;

    float distance(const std::vector<float>& a, const std::vector<float>& b);
    int findNearestCentroid(const std::vector<float>& point);
};

#endif // IVF_H
