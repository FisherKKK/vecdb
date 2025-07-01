#include "ivf.h"

IVF::IVF(int dim, int nlist) : dim_(dim), nlist_(nlist) {
    inverted_lists_.resize(nlist_);
}

float IVF::distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0;
    for (int i = 0; i < dim_; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}

void IVF::train(const std::vector<std::vector<float>>& data) {
    // K-means for centroids (simplified)
    std::vector<int> assignments(data.size());
    centroids_.resize(nlist_);
    for (int i = 0; i < nlist_; ++i) {
        centroids_[i] = data[i];
    }

    for (int iter = 0; iter < 10; ++iter) {
        // Assign
        for (size_t i = 0; i < data.size(); ++i) {
            float min_dist = -1;
            int best_centroid = -1;
            for (int j = 0; j < nlist_; ++j) {
                float dist = distance(data[i], centroids_[j]);
                if (best_centroid == -1 || dist < min_dist) {
                    min_dist = dist;
                    best_centroid = j;
                }
            }
            assignments[i] = best_centroid;
        }

        // Update
        std::vector<std::vector<float>> new_centroids(nlist_, std::vector<float>(dim_, 0.0f));
        std::vector<int> counts(nlist_, 0);
        for (size_t i = 0; i < data.size(); ++i) {
            int centroid_idx = assignments[i];
            for (int d = 0; d < dim_; ++d) {
                new_centroids[centroid_idx][d] += data[i][d];
            }
            counts[centroid_idx]++;
        }

        for (int i = 0; i < nlist_; ++i) {
            if (counts[i] > 0) {
                for (int d = 0; d < dim_; ++d) {
                    centroids_[i][d] = new_centroids[i][d] / counts[i];
                }
            }
        }
    }
}

int IVF::findNearestCentroid(const std::vector<float>& point) {
    float min_dist = -1;
    int best_centroid = -1;
    for (int i = 0; i < nlist_; ++i) {
        float dist = distance(point, centroids_[i]);
        if (best_centroid == -1 || dist < min_dist) {
            min_dist = dist;
            best_centroid = i;
        }
    }
    return best_centroid;
}

void IVF::add(const std::vector<float>& point) {
    int centroid_idx = findNearestCentroid(point);
    inverted_lists_[centroid_idx].push_back(original_index_counter_);
    original_points_.push_back(point);
    original_index_counter_++;
}

std::vector<std::pair<int, float>> IVF::search(const std::vector<float>& query, int k, int nprobe) {
    std::vector<std::pair<float, int>> centroid_distances;
    for (int i = 0; i < nlist_; ++i) {
        centroid_distances.push_back({distance(query, centroids_[i]), i});
    }
    std::sort(centroid_distances.begin(), centroid_distances.end());

    std::vector<std::pair<int, float>> results;
    for (int i = 0; i < nprobe; ++i) {
        int centroid_idx = centroid_distances[i].second;
        for (int point_idx : inverted_lists_[centroid_idx]) {
            results.push_back({point_idx, distance(query, original_points_[point_idx])});
        }
    }

    std::sort(results.begin(), results.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second < b.second;
    });

    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}
