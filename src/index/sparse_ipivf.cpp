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
        n_postings = doc_prune_strategy_.parameters.fixedSize.n_postings;
        fixed_pruning(word_map, n_postings);
    } else if (doc_prune_strategy_.type == DocPruneStrategyType::GlobalPrune) {
        n_postings = doc_prune_strategy_.parameters.fixedSize.n_postings;
        max_fraction = doc_prune_strategy_.parameters.globalPrune.fraction;
        global_pruning(word_map, n_postings);
        fixed_pruning(word_map, n_postings * max_fraction);
    }

    this->inverted_lists_ = new InvertedList[this->data_dim_];
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

#pragma omp parallel for
    for (uint32_t i = 0; i < this->data_dim_; ++i) {
        auto it = word_map.find(i);
        if (it != word_map.end()) {
            std::lock_guard<std::mutex> lock(ivf_mutex[i]);
            const auto& doc_infos = it->second;
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

#pragma omp parallel for
    for (int i = 0; i < query_num; ++i) {
        auto query_vector = query->GetSparseVectors()[i];
        this->search_one_query(query_vector, k, ids + i * k, dists + i * k);
    }

    return std::move(dataset_results);
}

void 
SparseIPIVF::multiply(std::vector<std::pair<uint32_t, float>> &query_pair, std::vector<float> &dists) const {
    for (uint32_t i = 0; i < query_pair.size(); ++i) {
        uint32_t term_id = query_pair[i].first;
        auto term_doc_num = this->inverted_lists_[term_id].doc_num_;

        if (term_doc_num == 0) {
            continue;
        }


        for (uint32_t j = 0; j < term_doc_num; ++j) {
            auto doc_id = this->inverted_lists_[term_id].ids_[j];
            auto value = this->inverted_lists_[term_id].vals_[j];
            dists[doc_id] += (-query_pair[i].second * value);
        }
    }
}

void 
SparseIPIVF::scan_sort(std::vector<float> &dists, 
                int64_t k,
                int64_t* res_ids,
                float* res_dists) const {

     MaxHeap heap(this->allocator_.get());
    float cur_heap_top = std::numeric_limits<float>::max();

    for (size_t i = 0; i < this->total_count_; ++i) {
        if(dists[i] == 0) {
            continue;
        }
        if (heap.size() < k || dists[i] < cur_heap_top) {
            heap.emplace(dists[i], i);
        }

        if (heap.size() > k) {
            heap.pop();
        }

        if (!heap.empty()) {
            cur_heap_top = heap.top().first;
        }
    }

    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        res_dists[j] = heap.top().first;
        res_ids[j] = heap.top().second;
        heap.pop();
    }
}

void
SparseIPIVF::search_one_query(const SparseVector& query_vector,
                              int64_t k,
                              int64_t* res_ids,
                              float* res_dists) const {
    std::vector<std::pair<uint32_t, float>> query_pair(query_vector.dim_);
    for (uint32_t i = 0; i < query_vector.dim_; ++i) {
        query_pair[i] = std::make_pair(query_vector.ids_[i], query_vector.vals_[i]);
    }

    if (query_cut_ > 0) {
        if (query_vector.dim_ > query_cut_) {
            std::nth_element(
                query_pair.begin(),
                query_pair.begin() + query_cut_,
                query_pair.end(),
                [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                    return a.second > b.second;
                });

            query_pair.resize(this->query_cut_);
        }
    }
                                
    std::vector<float> dists(this->total_count_, 0.0);

    multiply(query_pair, dists);
    scan_sort(dists, k, res_ids, res_dists);
}
}  // namespace vsag