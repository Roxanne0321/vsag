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

#include "sindi.h"
#if defined(ENABLE_AVX512)
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>
#include <random>

namespace vsag {
Sindi::Sindi(const SindiParameters& param,
                         const IndexCommonParam& index_common_param) {
    alpha_ = param.alpha;
    lambda_ = param.lambda;
    allocator_ = index_common_param.allocator_;
}

tl::expected<void, Error>
Sindi::serialize(std::ostream& out_stream) {
    out_stream.write(reinterpret_cast<const char*>(&total_count_), sizeof(total_count_));
    out_stream.write(reinterpret_cast<const char*>(&data_dim_), sizeof(data_dim_));
    out_stream.write(reinterpret_cast<const char*>(&lambda_), sizeof(lambda_));
    out_stream.write(reinterpret_cast<const char*>(&sigma_), sizeof(sigma_));

    for (uint32_t i = 0; i < data_dim_; ++i) {
        const InvertedList& list = inverted_lists_[i];

        out_stream.write(reinterpret_cast<const char*>(&list.doc_num_), sizeof(list.doc_num_));

        if (list.doc_num_ > 0) {
            out_stream.write(reinterpret_cast<const char*>(list.ids_),
                             list.doc_num_ * sizeof(uint32_t));
            out_stream.write(reinterpret_cast<const char*>(list.vals_),
                             list.doc_num_ * sizeof(float));
            out_stream.write(reinterpret_cast<const char*>(list.offsets_),
                             (sigma_ + 1) * sizeof(uint32_t));
        }
    }

    for (auto doc_id = 0; doc_id < total_count_; ++doc_id) {
        out_stream.write(reinterpret_cast<const char*>(&data_[doc_id].dim_), sizeof(uint32_t));

        if (data_[doc_id].dim_ > 0) {
            out_stream.write(reinterpret_cast<const char*>(data_[doc_id].ids_),
                             data_[doc_id].dim_ * sizeof(uint32_t));
            out_stream.write(reinterpret_cast<const char*>(data_[doc_id].vals_),
                             data_[doc_id].dim_ * sizeof(float));
        }
    }

    return {};
}

tl::expected<void, Error>
Sindi::deserialize(std::istream& in_stream) {
    in_stream.read(reinterpret_cast<char*>(&total_count_), sizeof(total_count_));
    in_stream.read(reinterpret_cast<char*>(&data_dim_), sizeof(data_dim_));
    in_stream.read(reinterpret_cast<char*>(&lambda_), sizeof(lambda_));
    in_stream.read(reinterpret_cast<char*>(&sigma_), sizeof(sigma_));

    inverted_lists_ = new InvertedList[data_dim_];

    for (uint32_t i = 0; i < data_dim_; ++i) {
        InvertedList& list = inverted_lists_[i];
        in_stream.read(reinterpret_cast<char*>(&list.doc_num_), sizeof(list.doc_num_));

        if (list.doc_num_ > 0) {
            list.ids_ = new uint32_t[list.doc_num_];
            list.vals_ = new float[list.doc_num_];
            list.offsets_ = new uint32_t[sigma_ + 1];
            in_stream.read(reinterpret_cast<char*>(list.ids_), list.doc_num_ * sizeof(uint32_t));
            in_stream.read(reinterpret_cast<char*>(list.vals_), list.doc_num_ * sizeof(float));
            in_stream.read(reinterpret_cast<char*>(list.offsets_),
                           (sigma_ + 1) * sizeof(uint32_t));
        }
    }

    data_ = new SparseVector[total_count_];
    for (auto doc_id = 0; doc_id < total_count_; ++doc_id) {
        in_stream.read(reinterpret_cast<char*>(&data_[doc_id].dim_), sizeof(uint32_t));

        if (data_[doc_id].dim_ > 0) {
            data_[doc_id].ids_ = new uint32_t[data_[doc_id].dim_];
            data_[doc_id].vals_ = new float[data_[doc_id].dim_];
            in_stream.read(reinterpret_cast<char*>(data_[doc_id].ids_),
                           data_[doc_id].dim_ * sizeof(uint32_t));
            in_stream.read(reinterpret_cast<char*>(data_[doc_id].vals_),
                           data_[doc_id].dim_ * sizeof(float));
        }
    }
    return {};
}

std::vector<int64_t>
Sindi::build(const DatasetPtr& base) {
    this->data_dim_ = 0;
    //// copy base dataset
    const SparseVector* sparse_ptr = base->GetSparseVectors();
    this->total_count_ = base->GetNumElements();
    this->data_ = new SparseVector[this->total_count_];
    for (size_t i = 0; i < this->total_count_; ++i) {
        const SparseVector& sv = sparse_ptr[i];
        for (uint32_t j = 1; j < sv.dim_; ++j) {
            assert(sv.ids_[j - 1] <= sv.ids_[j] && "IDs are not in ascending order");
        }

        if (sv.ids_[sv.dim_ - 1] > this->data_dim_) {
            this->data_dim_ = sv.ids_[sv.dim_ - 1];
        }

        this->data_[i].dim_ = sv.dim_;
        this->data_[i].ids_ = new uint32_t[this->data_[i].dim_];
        this->data_[i].vals_ = new float[this->data_[i].dim_];
        memcpy(this->data_[i].ids_, sv.ids_, this->data_[i].dim_ * sizeof(uint32_t));
        memcpy(this->data_[i].vals_, sv.vals_, this->data_[i].dim_ * sizeof(float));
    }

    this->data_dim_ += 1;

    ivf_mutex = std::vector<std::mutex>(this->data_dim_);
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>> word_map;

    vector_prune(word_map);

    build_inverted_lists(word_map);

    return {};
}

std::vector<uint32_t>
get_top_n_indices(const SparseVector& vec, float alpha) {
    float total_mass = 0.0f;
    std::vector<uint32_t> indices(vec.dim_);
    for (uint32_t i = 0; i < vec.dim_; ++i) {
        indices[i] = i;
        total_mass += vec.vals_[i];
    }

    if (alpha == 1) {
        return indices;
    }
    std::sort(
        indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
            return vec.vals_[a] > vec.vals_[b];
        });

    float part_mass = total_mass * alpha;
    float temp_mass = 0.0f;
    int max_index = 0;
    while(temp_mass < part_mass) {
        temp_mass += vec.vals_[indices[max_index]];
        max_index ++;
    }

    indices.resize(max_index);

    return indices;
}

void
Sindi::vector_prune(std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map) {
    for (size_t i = 0; i < this->total_count_; ++i) {
        const SparseVector& sv = data_[i];
        std::vector<uint32_t> top_n_indices = get_top_n_indices(sv, alpha_);
        for (auto j = 0; j < top_n_indices.size(); j++) {
            uint32_t word_id = sv.ids_[top_n_indices[j]];
            float val = sv.vals_[top_n_indices[j]];
            word_map[word_id].emplace_back(i, val);
        }
    }
}

void
Sindi::build_inverted_lists(
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map) {
    this->inverted_lists_ = new InvertedList[this->data_dim_];
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    sigma_ = total_count_ / lambda_ + ((total_count_ % lambda_) == 0 ? 0 : 1);

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
            this->inverted_lists_[i].offsets_ = new uint32_t[sigma_ + 1];
            this->inverted_lists_[i].offsets_[0] = 0;
            for (uint32_t j = 0; j < doc_num; j++) {
                this->inverted_lists_[i].ids_[j] = doc_infos[j].first;
                this->inverted_lists_[i].vals_[j] = doc_infos[j].second;
            }
            uint32_t doc_count = 0;
            for (uint32_t window_index = 1; window_index <= sigma_; ++window_index) {
                uint32_t next_window_boundary = window_index * lambda_;
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
Sindi::knn_search(const DatasetPtr& query,
                        int64_t k,
                        const std::string& parameters,
                        const std::function<bool(int64_t)>& filter) const {
    auto params = SindiSearchParameters::FromJson(parameters);
    this->num_threads_ = params.num_threads;
    this->beta_ = params.beta;
    this->gamma_ = params.gamma;

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
        std::vector<float> win_dists(lambda_, 0.0);
        this->search_one_query(query_vector, k, ids + i * k, dists + i * k, win_dists);
    }

    return std::move(dataset_results);
}

void
Sindi::search_one_query(const SparseVector& query_vector,
                              int64_t k,
                              int64_t* res_ids,
                              float* res_dists,
                              std::vector<float>& win_dists) const {
    int n = query_vector.dim_ * beta_;
    std::vector<std::pair<uint32_t, float>> elements(query_vector.dim_);
    std::vector<float> query_dense(data_dim_);

    for (uint32_t i = 0; i < query_vector.dim_; ++i) {
        elements[i] = {query_vector.ids_[i], query_vector.vals_[i]};
        query_dense[query_vector.ids_[i]] = query_vector.vals_[i];
    }

    std::nth_element(elements.begin(),
                     elements.begin() + n,
                     elements.end(),
                     [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                         return a.second > b.second;
                     });

    elements.resize(n);

    MaxHeap heap(this->allocator_.get());

    accumulation_scan(elements, heap, win_dists);

    reorder(query_dense, heap, k, res_ids, res_dists);
}

void
Sindi::accumulation_scan(std::vector<std::pair<uint32_t, float>>& query_vector,
                               MaxHeap& heap,
                               std::vector<float>& dists) const {
    float cur_heap_top = std::numeric_limits<float>::max();

    for (auto window_index = 0; window_index < sigma_; ++window_index) {
        uint32_t start = window_index * lambda_;
        for (auto term_index = 0; term_index < query_vector.size(); term_index++) {
            float query_val = -query_vector[term_index].second;
            auto term_id = query_vector[term_index].first;
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

        for (auto term_index = 0; term_index < query_vector.size(); term_index++) {
            auto term_id = query_vector[term_index].first;
            const InvertedList& list = inverted_lists_[term_id];
            if (list.doc_num_ == 0) [[unlikely]] {
                continue;
            }
            for (auto doc_id_index = list.offsets_[window_index];
                 doc_id_index < list.offsets_[window_index + 1];
                 ++doc_id_index) {
                auto doc_id = list.ids_[doc_id_index];
                auto temp_id = doc_id - start;
                if (dists[temp_id] >= cur_heap_top) [[likely]] {
                    dists[temp_id] = 0;
                    continue;
                } else {
                    heap.emplace(dists[temp_id], doc_id);
                }
                if (heap.size() > gamma_) {
                    heap.pop();
                }
                cur_heap_top = heap.top().first;
                dists[temp_id] = 0;
            }
        }
    }
}

void
Sindi::reorder(const std::vector<float>& query_dense,
                     MaxHeap& heap,
                     int64_t k,
                     int64_t* res_ids,
                     float* res_dists) const {
    MaxHeap final_heap(this->allocator_.get());
    float cur_heap_top = std::numeric_limits<float>::max();

    while (!heap.empty()) {
        auto doc_id = heap.top().second;

        const SparseVector& base_vector = data_[doc_id];

        float dist = DenseComputeIP(query_dense, base_vector);

        if (dist < cur_heap_top or final_heap.size() < k) {
            final_heap.emplace(dist, doc_id);
        }
        if (final_heap.size() > k) {
            final_heap.pop();
        }
        cur_heap_top = final_heap.top().first;
        heap.pop();
    }

    for (auto j = static_cast<int64_t>(final_heap.size() - 1); j >= 0; --j) {
        res_dists[j] = -final_heap.top().first;
        res_ids[j] = final_heap.top().second;
        final_heap.pop();
    }
}
}  // namespace vsag
