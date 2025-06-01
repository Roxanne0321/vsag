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
    build_strategy_ = param.build_strategy;
    window_size_ = param.window_size;
    allocator_ = index_common_param.allocator_;
}

tl::expected<void, Error>
SparseIPIVF::serialize(std::ostream& out_stream) {
    out_stream.write(reinterpret_cast<const char*>(&total_count_), sizeof(total_count_));
    out_stream.write(reinterpret_cast<const char*>(&data_dim_), sizeof(data_dim_));
    out_stream.write(reinterpret_cast<const char*>(&window_size_), sizeof(window_size_));
    out_stream.write(reinterpret_cast<const char*>(&window_num_), sizeof(window_num_));

    for (uint32_t i = 0; i < data_dim_; ++i) {
        const InvertedList& list = inverted_lists_[i];

        out_stream.write(reinterpret_cast<const char*>(&list.doc_num_), sizeof(list.doc_num_));

        if (list.doc_num_ > 0) {
            out_stream.write(reinterpret_cast<const char*>(list.ids_),
                             list.doc_num_ * sizeof(uint32_t));
            out_stream.write(reinterpret_cast<const char*>(list.vals_),
                             list.doc_num_ * sizeof(float));
            out_stream.write(reinterpret_cast<const char*>(list.offsets_),
                             (window_num_ + 1) * sizeof(uint32_t));
        }
    }

    return {};
}

tl::expected<void, Error>
SparseIPIVF::deserialize(std::istream& in_stream) {
    in_stream.read(reinterpret_cast<char*>(&total_count_), sizeof(total_count_));
    in_stream.read(reinterpret_cast<char*>(&data_dim_), sizeof(data_dim_));
    in_stream.read(reinterpret_cast<char*>(&window_size_), sizeof(window_size_));
    in_stream.read(reinterpret_cast<char*>(&window_num_), sizeof(window_num_));

    inverted_lists_ = new InvertedList[data_dim_];

    for (uint32_t i = 0; i < data_dim_; ++i) {
        InvertedList& list = inverted_lists_[i];
        in_stream.read(reinterpret_cast<char*>(&list.doc_num_), sizeof(list.doc_num_));

        if (list.doc_num_ > 0) {
            list.ids_ = new uint32_t[list.doc_num_];
            list.vals_ = new float[list.doc_num_];
            list.offsets_ = new uint32_t[window_num_ + 1];
            in_stream.read(reinterpret_cast<char*>(list.ids_), list.doc_num_ * sizeof(uint32_t));
            in_stream.read(reinterpret_cast<char*>(list.vals_), list.doc_num_ * sizeof(float));
            in_stream.read(reinterpret_cast<char*>(list.offsets_),
                           (window_num_ + 1) * sizeof(uint32_t));
        }
    }
    return {};
}

std::vector<int64_t>
SparseIPIVF::build(const DatasetPtr& base) {
    this->data_dim_ = 0;

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

    vector_prune(sparse_ptr, word_map);

    list_prune(word_map);

    build_inverted_lists(word_map);

    return {};
}

void
SparseIPIVF::vector_prune(
    const SparseVector* sparse_ptr,
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map) {
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
}

void
SparseIPIVF::list_prune(
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map) {
    if (doc_prune_strategy_.type == DocPruneStrategyType::FixedSize) {
        fixed_pruning(
            word_map, doc_prune_strategy_.parameters.fixedSize.n_postings, this->data_dim_);
    } else if (doc_prune_strategy_.type == DocPruneStrategyType::GlobalPrune) {
        global_pruning(
            word_map, doc_prune_strategy_.parameters.globalPrune.n_postings, this->data_dim_);
        fixed_pruning(word_map,
                      doc_prune_strategy_.parameters.globalPrune.n_postings *
                          doc_prune_strategy_.parameters.globalPrune.fraction,
                      this->data_dim_);
    }
}

void
SparseIPIVF::build_inverted_lists(
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map) {
    this->inverted_lists_ = new InvertedList[this->data_dim_];
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    window_num_ = total_count_ / window_size_ + ((total_count_ % window_size_) == 0 ? 0 : 1);

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
            this->inverted_lists_[i].offsets_ = new uint32_t[window_num_ + 1];
            this->inverted_lists_[i].offsets_[0] = 0;
            for (uint32_t j = 0; j < doc_num; j++) {
                this->inverted_lists_[i].ids_[j] = doc_infos[j].first;
                this->inverted_lists_[i].vals_[j] = doc_infos[j].second;
            }
            uint32_t doc_count = 0;
            for (uint32_t window_index = 1; window_index <= window_num_; ++window_index) {
                uint32_t next_window_boundary = window_index * window_size_;
                while (doc_count < doc_num &&
                       this->inverted_lists_[i].ids_[doc_count] < next_window_boundary) {
                    doc_count++;
                }
                this->inverted_lists_[i].offsets_[window_index] = doc_count;
            }
        }
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
        //std::cout << "query " << i << std::endl;
        std::vector<float> win_dists(window_size_, 0.0);
        this->search_one_query(query_vector,
                               k,
                               ids + i * k,
                               dists + i * k,
                               win_dists);
    }

    return std::move(dataset_results);
}

void
SparseIPIVF::search_one_query(const SparseVector& query_vector,
                              int64_t k,
                              int64_t* res_ids,
                              float* res_dists,
                              std::vector<float>& win_dists) const {
    accumulation_scan(
        query_vector, win_dists, k, res_ids, res_dists);
}

void
SparseIPIVF::accumulation_scan(const SparseVector& query_vector,
                               std::vector<float>& dists,
                               //std::vector<std::vector<float>>& product,
                               int64_t k,
                               int64_t* res_ids,
                               float* res_dists) const {
    MaxHeap heap(this->allocator_.get());
    float cur_heap_top = std::numeric_limits<float>::max();
    // std::unordered_set<uint32_t> heap_ids(k);

    for (auto window_index = 0; window_index < window_num_; ++window_index) {
        // auto start_time_1 = std::chrono::high_resolution_clock::now();
        uint32_t start = window_index * window_size_;
        for (auto term_index = 0; term_index < query_vector.dim_; term_index++) {
            float query_val = -query_vector.vals_[term_index];
            auto term_id = query_vector.ids_[term_index];
            const InvertedList& list = inverted_lists_[term_id];
            if (list.doc_num_ == 0) [[unlikely]] {
                continue;
            }
            for (auto doc_id_index = list.offsets_[window_index];
                 doc_id_index < list.offsets_[window_index + 1];
                 ++doc_id_index) {
                auto doc_id = list.ids_[doc_id_index];
                dists[doc_id - start] += list.vals_[doc_id_index] * query_val;
            }
        }

        // auto end_time_1 = std::chrono::high_resolution_clock::now();
        // accumulation_time +=
        //     std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_1 - start_time_1).count();

        // auto start_time_2 = std::chrono::high_resolution_clock::now();

        // for (auto term_index = 0; term_index < query_vector.dim_; term_index++) {
        //     auto term_id = query_vector.ids_[term_index];
        //     const InvertedList& list = inverted_lists_[term_id];
        //     if (list.doc_num_ == 0) [[unlikely]] {
        //         continue;
        //     }
        //     for (auto doc_id_index = list.offsets_[window_index];
        //          doc_id_index < list.offsets_[window_index + 1];
        //          ++doc_id_index) {
        //         auto doc_id = list.ids_[doc_id_index];
        //         auto temp_id = doc_id - start;
        //         if (dists[temp_id] < cur_heap_top or heap.size() < k) {
        //             if (heap_ids.find(doc_id) == heap_ids.end()) {  //没找到
        //                 heap_ids.insert(doc_id);
        //                 heap.emplace(dists[temp_id], doc_id);
        //             }
        //         }
        //         if (heap.size() > k) {
        //             heap.pop();
        //             heap_ids.erase(heap.top().first);
        //         }
        //         cur_heap_top = heap.top().first;
        //         dists[temp_id] = 0;
        //     }
        // }

        for (auto i = 0; i < window_size_; ++i) { // 最后一个window size
            // dists[i + start]入堆
            if (dists[i] >= cur_heap_top) [[likely]] {
                dists[i] = 0;
                continue;
            } else {
                heap.emplace(dists[i], i + start);
            }
            if(heap.size() > k) {
                heap.pop();
            }
            cur_heap_top = heap.top().first;
            dists[i] = 0;
        }
    }

    // auto end_time_2 = std::chrono::high_resolution_clock::now();
    // scan_time +=
    //     std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_2 - start_time_2).count();
    // }

    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        res_dists[j] = -heap.top().first;
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