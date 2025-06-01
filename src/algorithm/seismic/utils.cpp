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
              int n_postings, uint32_t dim) {
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
               int n_postings, uint32_t dim) {
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
}