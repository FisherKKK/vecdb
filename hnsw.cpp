#include "hnsw.h"
#include <queue>

HNSW::HNSW(int dim, int m, int ef_construction)
    : dim_(dim), m_(m), ef_construction_(ef_construction), max_level_(-1), entry_point_(-1), rng_(std::random_device{}()) {}

float HNSW::distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0;
    for (int i = 0; i < dim_; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}

int HNSW::getRandomLevel() {
    std::uniform_real_distribution<> dist(0, 1);
    int level = 0;
    while (dist(rng_) < 0.5) {
        level++;
    }
    return level;
}

std::vector<std::pair<int, float>> HNSW::searchLayer(const std::vector<float>& query, int entry_point, int ef, int level) {
    std::vector<std::pair<int, float>> result;
    if (entry_point == -1) {
        return result;
    }

    std::vector<bool> visited(nodes_.size(), false);
    std::priority_queue<std::pair<float, int>> candidates;
    candidates.push({-distance(query, nodes_[entry_point].point), entry_point});

    std::priority_queue<std::pair<float, int>> top_k;
    top_k.push({distance(query, nodes_[entry_point].point), entry_point});
    visited[entry_point] = true;

    while (!candidates.empty()) {
        auto curr = candidates.top();
        candidates.pop();
        int curr_node_idx = curr.second;

        if (top_k.top().first < -curr.first) {
            break;
        }

        for (int neighbor_idx : nodes_[curr_node_idx].connections[level]) {
            if (!visited[neighbor_idx]) {
                visited[neighbor_idx] = true;
                float dist = distance(query, nodes_[neighbor_idx].point);
                if (top_k.size() < ef || dist < top_k.top().first) {
                    candidates.push({-dist, neighbor_idx});
                    top_k.push({dist, neighbor_idx});
                    if (top_k.size() > ef) {
                        top_k.pop();
                    }
                }
            }
        }
    }

    while (!top_k.empty()) {
        result.push_back(std::make_pair(top_k.top().second, top_k.top().first));
        top_k.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

void HNSW::addPoint(const std::vector<float>& point) {
    int new_node_idx = nodes_.size();
    int level = getRandomLevel();

    Node new_node;
    new_node.point = point;
    new_node.connections.resize(level + 1);
    nodes_.push_back(new_node);

    if (entry_point_ == -1) {
        entry_point_ = new_node_idx;
        max_level_ = level;
        return;
    }

    int curr_node_idx = entry_point_;
    for (int l = max_level_; l > level; --l) {
        curr_node_idx = searchLayer(point, curr_node_idx, 1, l)[0].first;
    }

    for (int l = std::min(level, max_level_); l >= 0; --l) {
        auto neighbors = searchLayer(point, curr_node_idx, ef_construction_, l);
        std::vector<int> new_connections;
        for (const auto& neighbor : neighbors) {
            new_connections.push_back(neighbor.first);
        }

        // Connect new node to neighbors
        nodes_[new_node_idx].connections[l] = new_connections;

        // Connect neighbors to new node
        for (int neighbor_idx : new_connections) {
            nodes_[neighbor_idx].connections[l].push_back(new_node_idx);
            // Prune connections if necessary
            if (nodes_[neighbor_idx].connections[l].size() > m_) {
                // Simple pruning for demonstration
                nodes_[neighbor_idx].connections[l].erase(nodes_[neighbor_idx].connections[l].begin());
            }
        }
        curr_node_idx = neighbors[0].first;
    }

    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = new_node_idx;
    }
}

std::vector<std::pair<int, float>> HNSW::search(const std::vector<float>& query, int k) {
    int curr_node_idx = entry_point_;
    for (int l = max_level_; l > 0; --l) {
        curr_node_idx = searchLayer(query, curr_node_idx, 1, l)[0].first;
    }
    auto result = searchLayer(query, curr_node_idx, k, 0);
    if (result.size() > k) {
        result.resize(k);
    }
    return result;
}
