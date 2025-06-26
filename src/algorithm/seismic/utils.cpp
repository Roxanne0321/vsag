#include "algorithm/seismic/utils.h"

namespace vsag {
std::vector<uint32_t>
get_top_n_indices(const SparseVector& vec, uint32_t n) {
    std::vector<uint32_t> indices(vec.dim_);
    for (uint32_t i = 0; i < vec.dim_; ++i) {
        indices[i] = i;
    }
    if (n >= vec.dim_) {
        return indices;
    }

    std::nth_element(
        indices.begin(), indices.begin() + n, indices.end(), [&](uint32_t a, uint32_t b) {
            return vec.vals_[a] > vec.vals_[b];
        });

    return indices;
}

void
fixed_pruning(std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map,
              int n_postings,
              uint32_t dim) {
    for (uint32_t i = 0; i < dim; ++i) {
        if (word_map.find(i) != word_map.end()) {
            auto& doc_infos = word_map[i];

            std::sort(doc_infos.begin(),
                      doc_infos.end(),
                      [](const std::pair<uint32_t, float> a, const std::pair<uint32_t, float> b) {
                          return a.second > b.second;
                      });

            if (doc_infos.size() > n_postings) {
                doc_infos.resize(n_postings);
            }

            word_map[i] = doc_infos;
        }
    }
}

void
global_pruning(std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map,
               int n_postings,
               uint32_t dim) {
    size_t total_postings = dim * n_postings;

    std::vector<std::tuple<float, uint32_t, uint32_t>> postings;

    for (const auto& kv : word_map) {
        uint32_t word_id = kv.first;
        for (const auto& doc_info : kv.second) {
            postings.emplace_back(doc_info.second, doc_info.first, word_id);
        }
    }

    total_postings = std::min(total_postings, postings.size());

    std::nth_element(postings.begin(),
                     postings.begin() + total_postings,
                     postings.end(),
                     [](const std::tuple<float, uint32_t, uint32_t>& a,
                        const std::tuple<float, uint32_t, uint32_t>& b) {
                         return std::get<0>(a) > std::get<0>(b);
                     });

    for (auto& kv : word_map) {
        kv.second.clear();
    }

    for (auto it = postings.begin(); it != postings.begin() + total_postings; ++it) {
        float score = std::get<0>(*it);
        uint32_t docid = std::get<1>(*it);
        uint32_t word_id = std::get<2>(*it);

        word_map[word_id].emplace_back(docid, score);
    }
}

float
SparseComputeIP(const SparseVector& sv1, const SparseVector& sv2) {
    float sum = 0.0f;
    int i = 0, j = 0;

    while (i < sv1.dim_ && j < sv2.dim_) {
        if (sv1.ids_[i] == sv2.ids_[j]) {
            sum += sv1.vals_[i] * sv2.vals_[j];
            i++;
            j++;
        } else if (sv1.ids_[i] < sv2.ids_[j]) {
            i++;
        } else {
            j++;
        }
    }
    return -sum;
}

float DenseComputeIP(const std::vector<float> &query, const SparseVector& base) {
    const size_t N_LANES = 4;
    float result[N_LANES] = {0.0, 0.0, 0.0, 0.0};
    
    // 处理完整的N_LANES块
    size_t full_blocks = base.dim_ / N_LANES;
    
    for (size_t i = 0; i < full_blocks * N_LANES; i += N_LANES) {
        result[0] += query[base.ids_[i]] * base.vals_[i];
        result[1] += query[base.ids_[i + 1]] * base.vals_[i + 1];
        result[2] += query[base.ids_[i + 2]] * base.vals_[i + 2];
        result[3] += query[base.ids_[i + 3]] * base.vals_[i + 3];
    }

    // 处理剩余部分
    for (size_t i = full_blocks * N_LANES; i < base.dim_; ++i) {
        result[0] += query[base.ids_[i]] * base.vals_[i];
    }

    // 汇总结果
    return - (result[0] + result[1] + result[2] + result[3]);
}

// 该函数没有删除数目为零的簇
void
initialize_kmeans(const SparseVector* data,
                  std::vector<uint32_t>& doc_ids,
                  std::vector<std::vector<uint32_t>>& clusters,
                  uint32_t n_centroids,
                  uint32_t min_cluster_size) {
    std::vector<uint32_t> centroid_ids(n_centroids);
    // random choose n centroids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(doc_ids.begin(), doc_ids.end(), gen);
    centroid_ids.assign(doc_ids.begin(), doc_ids.begin() + n_centroids);

#pragma omp parallel for
    for (auto doc_id : doc_ids) {
        SparseVector doc_vector = data[doc_id];
        int argmin = 0;
        float min = std::numeric_limits<float>::max();

        for (int i = 0; i < n_centroids; i++) {
            auto cen_id = centroid_ids[i];
            SparseVector cen_vector = data[centroid_ids[i]];
            float dist = SparseComputeIP(doc_vector, cen_vector);
            if (dist < min) {
                argmin = i;
                min = dist;
            }
        }
#pragma omp critical
        {
            clusters[argmin].emplace_back(doc_id);
        }
    }

    std::vector<uint32_t> to_be_replaced;

    for (int i = 0; i < n_centroids; i++) {
        if (clusters[i].size() > 0 && clusters[i].size() < min_cluster_size) {
            to_be_replaced.insert(to_be_replaced.end(), clusters[i].begin(), clusters[i].end());
            clusters[i].clear();
        }
    }

    for (auto doc_id : to_be_replaced) {
        SparseVector doc_vector = data[doc_id];
        int argmin = 0;
        float min = std::numeric_limits<float>::max();

        for (int i = 0; i < n_centroids; ++i) {
            if (clusters[i].empty()) {
                continue;
            }
            SparseVector cen_vector = data[centroid_ids[i]];
            float dist = SparseComputeIP(doc_vector, cen_vector);
            if (dist < min) {
                argmin = i;
                min = dist;
            }
        }
        clusters[argmin].emplace_back(doc_id);
    }
}

void
update_cluster_centers(const SparseVector* data,
                       std::vector<std::vector<uint32_t>>& clusters,
                       std::vector<uint32_t>& centroids_ids,
                       uint32_t n_centroids) {
    for (auto cluster_id = 0; cluster_id < n_centroids; ++cluster_id) {
        const auto& cluster = clusters[cluster_id];
        SparseVector mean_vector;
        std::unordered_map<uint32_t, std::pair<uint32_t, float>>
            dim_sums;  // 存储每个维度取值累加和，以及取值个数

        for (auto doc_id : cluster) {
            SparseVector sv = data[doc_id];
            for (auto i = 0; i < sv.dim_; ++i) {
                dim_sums[sv.ids_[i]].second += sv.vals_[i];
                dim_sums[sv.ids_[i]].first++;
            }
        }

        //如果迭代时间太长可以采用截断
        std::vector<std::pair<uint32_t, float>> sorted_dim_sums;
        sorted_dim_sums.reserve(dim_sums.size());
        for (const auto& [dimension, value_count_sum] : dim_sums) {
            uint32_t count = value_count_sum.first;
            float sum = value_count_sum.second;
            float average = sum / static_cast<float>(count);
            sorted_dim_sums.emplace_back(dimension, average);
        }

        std::sort(sorted_dim_sums.begin(),
                  sorted_dim_sums.end(),
                  [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                      return a.first < b.first;
                  });

        mean_vector.dim_ = dim_sums.size();
        mean_vector.ids_ = new uint32_t[dim_sums.size()];
        mean_vector.vals_ = new float[dim_sums.size()];

        for (auto i = 0; i < dim_sums.size(); ++i) {
            mean_vector.ids_[i] = sorted_dim_sums[i].first;
            mean_vector.vals_[i] = sorted_dim_sums[i].second;
        }

        float min_distance = std::numeric_limits<float>::max();
        uint32_t best_doc_id = 0;

        // 选择和均值向量内积最大的数据点作为簇心
        for (auto doc_id : cluster) {
            SparseVector sv = data[doc_id];
            float distance = SparseComputeIP(sv, mean_vector);
            if (distance < min_distance) {
                min_distance = distance;
                best_doc_id = doc_id;
            }
        }
        centroids_ids[cluster_id] = best_doc_id;
        std::cout << "best doc id: " << best_doc_id << std::endl;
    }
}

void
do_kmeans_on_doc_id(const SparseVector* data,
                    std::vector<uint32_t>& doc_ids,
                    std::vector<std::vector<uint32_t>>& clusters,
                    uint32_t n_centroids,
                    uint32_t min_cluster_size,
                    uint32_t k) {
    std::vector<uint32_t> centroid_ids(n_centroids);

    initialize_kmeans(data, doc_ids, clusters, n_centroids, min_cluster_size);

    for (auto iter = 1; iter < k; ++iter) {
        update_cluster_centers(data, clusters, centroid_ids, n_centroids);
        for(auto cluster_id = 0; cluster_id < n_centroids; ++cluster_id) {
            // std::cout << "centroid id: " << centroid_ids[cluster_id] << std::endl;
            clusters[cluster_id].clear();
        }
#pragma omp parallel for
        for (auto doc_id : doc_ids) {
            SparseVector doc_vector = data[doc_id];
            int argmin = 0;
            float min = std::numeric_limits<float>::max();

            for (int i = 0; i < n_centroids; i++) {
                auto cen_id = centroid_ids[i];
                SparseVector cen_vector = data[centroid_ids[i]];
                float dist = SparseComputeIP(doc_vector, cen_vector);
                if (dist < min) {
                    argmin = i;
                    min = dist;
                }
            }
#pragma omp critical
            {
                clusters[argmin].emplace_back(doc_id);
            }
        }
    }
}

void
random_kmeans(const SparseVector* data,
              std::vector<uint32_t>& doc_ids,
              std::vector<std::vector<uint32_t>>& clusters,
              uint32_t n_centroids,
              uint32_t min_cluster_size) {
    std::vector<uint32_t> centroid_ids(n_centroids);

    //// random choose n centroids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(doc_ids.begin(), doc_ids.end(), gen);
    centroid_ids.assign(doc_ids.begin(), doc_ids.begin() + n_centroids);

    for (size_t i = 0; i < doc_ids.size(); ++i) {
        clusters[i % n_centroids].push_back(doc_ids[i]);
    }
}

void
energy_preserving_summary(const SparseVector* data,
                          std::vector<uint32_t>& ids,
                          std::vector<float>& vals,
                          std::vector<uint32_t> cluster,
                          float fraction) {
    std::unordered_map<uint32_t, float> hash;
    for (auto doc_id : cluster) {
        SparseVector sv = data[doc_id];
        for (uint32_t i = 0; i < sv.dim_; ++i) {
            auto it = hash.find(sv.ids_[i]);
            if (it != hash.end()) {
                if (it->second < sv.vals_[i]) {
                    it->second = sv.vals_[i];
                }
            } else {
                hash[sv.ids_[i]] = sv.vals_[i];
            }
        }
    }

    std::vector<std::pair<uint32_t, float>> components_values(hash.begin(), hash.end());

    std::sort(components_values.begin(), components_values.end(), [](const auto& a, const auto& b) {
        return b.second < a.second;
    });

    float total_sum = std::accumulate(
        components_values.begin(), components_values.end(), 0.0f, [](float sum, const auto& pair) {
            return sum + static_cast<float>(pair.second);  // Assume T can be casted to float
        });

    float acc = 0.0f;
    for (const auto& [tid, v] : components_values) {
        acc += v;
        ids.emplace_back(tid);
        if (acc / total_sum > fraction) {
            break;
        }
    }

    std::sort(ids.begin(), ids.end());

    for (auto tid : ids) {
        vals.emplace_back(hash[tid]);
    }
}
}  // namespace vsag