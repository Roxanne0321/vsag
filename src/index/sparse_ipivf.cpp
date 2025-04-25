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

tl::expected<void, Error>
SparseIPIVF::serialize(std::ostream& out_stream) {
    out_stream.write(reinterpret_cast<const char*>(&total_count_), sizeof(total_count_));
    out_stream.write(reinterpret_cast<const char*>(&data_dim_), sizeof(data_dim_));

    for (uint32_t i = 0; i < data_dim_; ++i) {
        const InvertedList& list = inverted_lists_[i];

        out_stream.write(reinterpret_cast<const char*>(&list.doc_num_), sizeof(list.doc_num_));

        if (list.doc_num_ > 0) {
            out_stream.write(reinterpret_cast<const char*>(list.ids_),
                             list.doc_num_ * sizeof(uint32_t));
            out_stream.write(reinterpret_cast<const char*>(list.vals_),
                             list.doc_num_ * sizeof(float));
        }
    }

    return {};
}

tl::expected<void, Error>
SparseIPIVF::deserialize(std::istream& in_stream) {
    in_stream.read(reinterpret_cast<char*>(&total_count_), sizeof(total_count_));
    in_stream.read(reinterpret_cast<char*>(&data_dim_), sizeof(data_dim_));

    inverted_lists_ = new InvertedList[data_dim_];

    for (uint32_t i = 0; i < data_dim_; ++i) {
        InvertedList& list = inverted_lists_[i];
        in_stream.read(reinterpret_cast<char*>(&list.doc_num_), sizeof(list.doc_num_));

        if (list.doc_num_ > 0) {
            list.ids_ = new uint32_t[list.doc_num_];
            list.vals_ = new float[list.doc_num_];
            in_stream.read(reinterpret_cast<char*>(list.ids_), list.doc_num_ * sizeof(uint32_t));
            in_stream.read(reinterpret_cast<char*>(list.vals_), list.doc_num_ * sizeof(float));
        }
    }
    return {};
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
    this->window_size_ = params.window_size;

    uint32_t query_num = query->GetNumElements();
    auto dataset_results = Dataset::Make();
    dataset_results->Dim(query_num * k)->NumElements(1)->Owner(true, allocator_.get());
    auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * query_num * k);
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator_->Allocate(sizeof(float) * query_num * k);
    dataset_results->Distances(dists);

    omp_set_num_threads(num_threads_);
    std::vector<float> win_dists(window_size_, 0.0);

    //uint32_t fp_cmp = 0;

#pragma omp parallel for
    for (int i = 0; i < query_num; ++i) {
        auto query_vector = query->GetSparseVectors()[i];
        //uint32_t temp_cmp;
        this->search_one_query(query_vector, k, ids + i * k, dists + i * k, win_dists);
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
                              float* res_dists,
                              std::vector<float>& win_dists) const {
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
    multiply(query_pair, product);
    accumulation_scan(query_pair, win_dists, product, k, res_ids, res_dists);
}

void
SparseIPIVF::accumulation_scan(std::vector<std::pair<uint32_t, float>>& query_pair,
                               std::vector<float>& dists,
                               std::vector<std::vector<float>>& product,
                               int64_t k,
                               int64_t* res_ids,
                               float* res_dists) const {
    MaxHeap heap(this->allocator_.get());
    float cur_heap_top = std::numeric_limits<float>::max();

    uint32_t start = UINT32_MAX;
    uint32_t next_start = UINT32_MAX;
    uint32_t max_doc_id = 0;
    size_t query_term_num = query_pair.size();
    std::vector<uint32_t> list_index(query_term_num, 0);

    for (auto i = 0; i < query_term_num; ++i) {
        auto term_id = query_pair[i].first;
        auto term_doc_num = this->inverted_lists_[term_id].doc_num_;
        if (term_doc_num == 0) {
            continue;
        }

        auto min_ = this->inverted_lists_[term_id].ids_[0];
        if (min_ < start) {
            start = min_;
        }

        auto max_ = this->inverted_lists_[term_id].ids_[term_doc_num - 1];
        if (max_ > max_doc_id) {
            max_doc_id = max_;
        }
    }

    while (start < max_doc_id) {
        for (auto term_index = 0; term_index < query_term_num; term_index++) {
            uint32_t doc_id_index = list_index[term_index];

            if (doc_id_index == -1) {
                continue;  //标志着这一列扫完了
            }
            auto term_id = query_pair[term_index].first;
            const InvertedList& list = inverted_lists_[term_id];
            for (; doc_id_index < product[term_index].size() &&
                   list.ids_[doc_id_index] < start + window_size_;
                 doc_id_index++) {
                auto doc_id = list.ids_[doc_id_index];
                dists[doc_id - start] += product[term_index][doc_id_index];
            }
            if (doc_id_index < product[term_index].size()) {
                list_index[term_index] = doc_id_index;
                auto doc_id = list.ids_[doc_id_index];
                if (doc_id < next_start) {
                    next_start = doc_id;
                }
            } else {
                list_index[term_index] = -1;
            }
        }

        for (auto i = 0; i < window_size_; ++i) {
            //dists[i + start]入堆
            if (heap.size() < k or dists[i] < cur_heap_top) {
                heap.emplace(dists[i], i + start);
            }
            if (heap.size() > k) {
                heap.pop();
            }
            cur_heap_top = heap.top().first;
            dists[i] = 0;
        }

        start = next_start;
        next_start = UINT32_MAX;
    }

    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        res_dists[j] = heap.top().first;
        res_ids[j] = heap.top().second;
        heap.pop();
    }
}

void
SparseIPIVF::multiply(std::vector<std::pair<uint32_t, float>>& query_pair,
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