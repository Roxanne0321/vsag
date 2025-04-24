// Copyright 2024-present the vsag pr
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sparse_ipivf.h"

#include <random>

namespace vsag {
SparseIPIVF::SparseIPIVF(const SparseIPIVFParameters& param,
                         const IndexCommonParam& index_common_param) {
    doc_prune_strategy_ = param.doc_prune_strategy;
    vector_prune_strategy_ = param.vector_prune_strategy;
    allocator_ = index_common_param.allocator_;
}

std::vector<uint32_t>
SparseIPIVF::get_top_n_indices(const SparseVector& vec, uint32_t n) {
    std::vector<uint32_t> indices(vec.dim_);
    for (uint32_t i = 0; i < vec.dim_; ++i) {
        indices[i] = i;
    }
    if (n >= vec.dim_) {
        return indices;
    }

    // 使用std::nth_element 找到第n个最大的值的位置
    std::nth_element(
        indices.begin(), indices.begin() + n, indices.end(), [&](uint32_t a, uint32_t b) {
            return vec.vals_[a] > vec.vals_[b];  // 降序比较
        });

    return indices;
}

std::vector<int64_t>
SparseIPIVF::build(const DatasetPtr& base) {
    this->data_dim_ = 0;
    //// copy base dataset
    const SparseVector* sparse_ptr = base->GetSparseVectors();
    this->total_count_ = base->GetNumElements();
    for (size_t i = 0; i < this->total_count_; ++i) {
        const SparseVector& sv = sparse_ptr[i];
        for (uint32_t j = 0; j < sv.dim_; ++j) {
            if (sv.ids_[j] > this->data_dim_) {
                this->data_dim_ = sv.ids_[j];
            }
        }
    }

    this->data_dim_ += 1;

    ivf_mutex = std::vector<std::mutex>(this->data_dim_);
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>> word_map;

    if (vector_prune_strategy_.type == VectorPruneStrategyType::VectorPrune) {
        int n_cut = vector_prune_strategy_.vectorPrune.n_cut;
        for (size_t i = 0; i < this->total_count_; ++i) {
            const SparseVector& sv = sparse_ptr[i];
            std::vector<uint32_t> top_n_indices = get_top_n_indices(sv, n_cut);
            for (auto j = 0; j < std::min(n_cut, static_cast<int>(sv.dim_)); j++) {
                uint32_t word_id = sv.ids_[top_n_indices[j]];
                float val = sv.vals_[top_n_indices[j]];
                word_map[word_id].emplace_back(i, val);
            }
        }
    } else if (vector_prune_strategy_.type == VectorPruneStrategyType::NotPrune) {
        for (size_t i = 0; i < this->total_count_; ++i) {
            const SparseVector& sv = sparse_ptr[i];
            for (uint32_t j = 0; j < sv.dim_; ++j) {
                uint32_t word_id = sv.ids_[j];
                float val = sv.vals_[j];
                word_map[word_id].emplace_back(i, val);
            }
        }
    }

    if (doc_prune_strategy_.type == DocPruneStrategyType::FixedSize) {
        fixed_pruning(word_map, doc_prune_strategy_.parameters.fixedSize.n_postings);
    } else if (doc_prune_strategy_.type == DocPruneStrategyType::GlobalPrune) {
        global_pruning(word_map, doc_prune_strategy_.parameters.globalPrune.n_postings);
        fixed_pruning(word_map,
                      doc_prune_strategy_.parameters.globalPrune.n_postings *
                          doc_prune_strategy_.parameters.globalPrune.fraction);
    }

    this->inverted_lists_ = new InvertedList[this->data_dim_];
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

#pragma omp parallel for
    for (uint32_t i = 0; i < this->data_dim_; ++i) {
        auto it = word_map.find(i);
        if (it != word_map.end()) {
            std::lock_guard<std::mutex> lock(ivf_mutex[i]);
            auto& doc_infos = it->second;
            std::sort(doc_infos.begin(),
                      doc_infos.end(),
                      [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                          return a.first < b.first;
                      });
            uint32_t doc_num = static_cast<uint32_t>(doc_infos.size());
            this->inverted_lists_[i].doc_num_ = doc_num;
            this->inverted_lists_[i].ids_ = new uint32_t[doc_num];
            this->inverted_lists_[i].vals_ = new float[doc_num];
            for (uint32_t j = 0; j < doc_num; j++) {
                this->inverted_lists_[i].ids_[j] = doc_infos[j].first;
                this->inverted_lists_[i].vals_[j] = doc_infos[j].second;
            }
        }
    }

    return {};
}

void
SparseIPIVF::fixed_pruning(
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map,
    int n_postings) {
    //int unique_term_ids = 0;
    for (uint32_t i = 0; i < this->data_dim_; ++i) {
        if (word_map.find(i) != word_map.end()) {
            //unique_term_ids ++;
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
SparseIPIVF::global_pruning(
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map,
    int n_postings) {
    // Calculate total postings to select
    size_t total_postings = this->data_dim_ * n_postings;  //seismic中是整个倒排列表的长度
    //std::cout << "total_postings: " << total_postings <<std::endl;

    // Collect all postings in a single vector with additional information
    std::vector<std::tuple<float, uint32_t, uint32_t>> postings;  // (score, docid, word_id)

    for (const auto& kv : word_map) {
        uint32_t word_id = kv.first;
        for (const auto& doc_info : kv.second) {
            postings.emplace_back(doc_info.second, doc_info.first, word_id);
        }
    }

    // Determine the actual number of postings to select
    total_postings = std::min(total_postings, postings.size());

    // Partially sort the postings to find the n-th largest element
    std::nth_element(postings.begin(),
                     postings.begin() + total_postings,
                     postings.end(),
                     [](const std::tuple<float, uint32_t, uint32_t>& a,
                        const std::tuple<float, uint32_t, uint32_t>& b) {
                         return std::get<0>(a) > std::get<0>(b);
                     });

    // Clear the word_map and add back the selected postings
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

DatasetPtr
SparseIPIVF::knn_search(const DatasetPtr& query,
                        int64_t k,
                        const std::string& parameters,
                        const std::function<bool(int64_t)>& filter) const {
    auto params = SparseIPIVFSearchParameters::FromJson(parameters);
    this->num_threads_ = params.num_threads;
    this->query_cut_ = params.query_cut;

    uint32_t query_num = query->GetNumElements();
    auto dataset_results = Dataset::Make();
    dataset_results->Dim(query_num * k)->NumElements(1)->Owner(true, allocator_.get());
    auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * query_num * k);
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator_->Allocate(sizeof(float) * query_num * k);
    dataset_results->Distances(dists);

    omp_set_num_threads(num_threads_);

    //uint32_t fp_cmp = 0;

#pragma omp parallel for
    for (int i = 0; i < query_num; ++i) {
        auto query_vector = query->GetSparseVectors()[i];
        //uint32_t temp_cmp;
        this->search_one_query(query_vector, k, ids + i * k, dists + i * k);
        // #pragma omp critical
        //     {
        //         fp_cmp += temp_cmp;
        //     }
    }

    //std::cout << "fp cmp: " << fp_cmp <<std::endl;
    return std::move(dataset_results);
}

void
SparseIPIVF::search_one_query(const SparseVector& query_vector,
                              int64_t k,
                              int64_t* res_ids,
                              float* res_dists) const {
    std::vector<std::pair<uint32_t, float>> query_pair;
    for (uint32_t i = 0; i < query_vector.dim_; ++i) {
        query_pair.emplace_back(query_vector.ids_[i], query_vector.vals_[i]);
    }

    if (query_cut_ > 0) {
        std::sort(query_pair.begin(),
                  query_pair.end(),
                  [](const std::pair<uint32_t, float> a, const std::pair<uint32_t, float> b) {
                      return a.second > b.second;
                  });

        if (query_vector.dim_ > query_cut_) {
            query_pair.resize(this->query_cut_);
        }
    }
    std::vector<std::vector<float>> product(query_pair.size());
    std::vector<float> dists(this->total_count_, 0.0);
    multiply(query_pair, product);
    accumulation_scan(query_pair, dists, product, k, res_ids, res_dists);
}

void
SparseIPIVF::accumulation_scan(std::vector<std::pair<uint32_t, float>> &query_pair,
                          std::vector<float>& dists,
                          std::vector<std::vector<float>>& product,
                          int64_t k,
                          int64_t* res_ids,
                          float* res_dists) const {
    MaxHeap heap(this->allocator_.get());
    float cur_heap_top = std::numeric_limits<float>::max();

    uint32_t last_min = UINT32_MAX;
    uint32_t cur_min = UINT32_MAX;
    //uint32_t cur_max = 0;
    uint32_t max_term_doc_num = 0;

    for (auto i = 0; i < query_pair.size(); ++i) {
        auto term_id = query_pair[i].first;
        auto term_doc_num = this->inverted_lists_[term_id].doc_num_;
        if (term_doc_num == 0) {
            continue;
        }

        if(term_doc_num > max_term_doc_num) {
            max_term_doc_num = term_doc_num;
        }

        auto cur_id = this->inverted_lists_[term_id].ids_[0];
        if (cur_id < last_min) {
            last_min = cur_id;
        }
    }

    // for(auto i = 0; i < 10; i++) {
    //     auto term_id = query_pair[i].first;
    //     auto term_doc_num = this->inverted_lists_[term_id].doc_num_;
    //     std::cout << term_id << std::endl;
    //     for(auto j = term_doc_num - 1; j >= term_doc_num - 10; j) {
    //         std::cout << this->inverted_lists_[term_id].ids_[j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    //std::vector<uint32_t> diff_min_max(max_term_doc_num);

    for (auto i = 0; i < max_term_doc_num; ++i) {
        for (auto term_index = 0; term_index < query_pair.size(); ++term_index) {
            auto term_id = query_pair[term_index].first;
            if (i < product[term_index].size()) {
                auto doc_id = this->inverted_lists_[term_id].ids_[i];
                dists[doc_id] += product[term_index][i];
                if (doc_id < cur_min) {
                    cur_min = doc_id;
                }
                // if (doc_id > cur_max) {
                //     cur_max = doc_id;
                // }
            }
        }
        for (auto j = last_min; j < cur_min; ++j) {
            if (dists[j] == 0) {
                continue;
            }
            if (heap.size() < k || dists[j] < cur_heap_top) {
                heap.emplace(dists[j], j);
            }

            if (heap.size() > k) {
                heap.pop();
            }

            if (!heap.empty()) {
                cur_heap_top = heap.top().first;
            }
        }
        //std::cout << cur_max - cur_min << " " << std::endl;;
        last_min = cur_min;
        cur_min = UINT32_MAX;
        //cur_max = 0;
    }
    //uint32_t avg_diff = 0;

    // for(const auto &diff : diff_min_max) {
    //     avg_diff += diff;
    // }

    //std::cout << "avg diff: " << static_cast<float>(avg_diff) / static_cast<float>(max_term_doc_num) << " ";

    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        res_dists[j] = heap.top().first;
        res_ids[j] = heap.top().second;
        heap.pop();
    }
}

void
SparseIPIVF::multiply(std::vector<std::pair<uint32_t, float>> &query_pair,
                      std::vector<std::vector<float>>& product) const {
    for (uint32_t i = 0; i < query_pair.size(); ++i) {
        uint32_t term_id = query_pair[i].first;
        auto term_doc_num = this->inverted_lists_[term_id].doc_num_;

        if (term_doc_num == 0) {
            continue;
        }

        product[i].resize(term_doc_num);

        float q_val = -query_pair[i].second;

        FP32ComputeSIP(
            &q_val, this->inverted_lists_[term_id].vals_, product[i].data(), term_doc_num);
    }
}
}  // namespace vsag