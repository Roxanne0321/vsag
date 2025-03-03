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

#include "sparse_ivf.h"

namespace vsag {
void
PrintSparseVector(const SparseVector& sv) {
    std::cout << "SparseVector:" << std::endl;
    std::cout << "Dimension: " << sv.dim_ << std::endl;
    std::cout << "IDs: ";
    for (int i = 0; i < sv.dim_; i++) {
        std::cout << sv.ids_[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Values: ";
    for (int i = 0; i < sv.dim_; i++) {
        std::cout << sv.vals_[i] << " ";
    }
    std::cout << std::endl;
}

static float
SparseComputeIP(const SparseVector& sv1, const SparseVector& sv2) {
    float sum = 0.0f;
    int i = 0, j = 0;

    while (i < sv1.dim_ && j < sv2.dim_) {
        if (sv1.ids_[i] == sv2.ids_[j]) {
            sum += sv1.vals_[i] * sv2.vals_[j];
            i++;
            j++;
        } else if (sv1.ids_[i] < sv2.ids_[j]) {
            // Increment pointer for the first vector
            i++;
        } else {
            // Increment pointer for the second vector
            j++;
        }
    }
    return -sum;
}

float
SparseComputeIPBruteForce(const SparseVector& sv1, const SparseVector& sv2) {
    float sum = 0.0f;

    for (size_t i = 0; i < sv1.dim_; ++i) {
        for (size_t j = 0; j < sv2.dim_; ++j) {
            if (sv1.ids_[i] == sv2.ids_[j]) {
                sum += sv1.vals_[i] * sv2.vals_[j];
            }
        }
    }
    return -sum;
}

SparseIVF::SparseIVF(const SparseIVFParameter& param, const IndexCommonParam& index_common_param) {
    allocator_ = index_common_param.allocator_;
}

std::vector<int64_t>
SparseIVF::build(const DatasetPtr& base) {
    return this->add(base);
}

std::vector<int64_t>
SparseIVF::add(const DatasetPtr& base) {
    const SparseVector* sparse_ptr = base->GetSparseVectors();
    this->total_count_ = base->GetNumElements();
    this->data_ = new SparseVector[this->total_count_];
    for(size_t i = 0; i < this->total_count_; ++i){
        this->data_[i].dim_ = sparse_ptr[i].dim_;
        this->data_[i].ids_ = new uint32_t[this->data_[i].dim_];
        this->data_[i].vals_ = new float[this->data_[i].dim_];
        memcpy(this->data_[i].ids_, sparse_ptr[i].ids_, this->data_[i].dim_ * sizeof(uint32_t));
        memcpy(this->data_[i].vals_, sparse_ptr[i].vals_, this->data_[i].dim_ * sizeof(float));
    }

    std::unordered_map<uint32_t, std::vector<uint32_t>> word_map;

    for (size_t i = 0; i < this->total_count_; ++i) {
        const SparseVector& sv = data_[i];
        for (uint32_t j = 1; j < sv.dim_; ++j) {
            assert(sv.ids_[j - 1] <= sv.ids_[j] && "IDs are not in ascending order");
        }

        for (uint32_t j = 0; j < sv.dim_; ++j) {
            uint32_t word_id = sv.ids_[j];

            if (word_id > this->data_dim_) {
                this->data_dim_ = word_id;
            }
            word_map[word_id].emplace_back(i);
        }
    }

    this->inverted_lists_ = new InvertedList[this->data_dim_ + 1];
    for (uint32_t i = 0; i <= this->data_dim_; ++i) {
        if (word_map.find(i) != word_map.end()) {
            auto& doc_infos = word_map[i];
            uint32_t doc_num = static_cast<uint32_t>(doc_infos.size());
            this->inverted_lists_[i].doc_num_ = doc_num;
            this->inverted_lists_[i].ids_ = new uint32_t[doc_num];
            //this->inverted_lists_[i].ids_ = (uint32_t*)allocator_->Allocate(sizeof(uint32_t) * doc_num);
            std::memcpy(this->inverted_lists_[i].ids_, doc_infos.data(), doc_num  * sizeof(uint32_t));
        }
    }
    return {};
}

DatasetPtr
SparseIVF::knn_search(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const std::function<bool(int64_t)>& filter) const {
    uint32_t query_num = query->GetNumElements();
    auto dataset_results = Dataset::Make();
    dataset_results->Dim(query_num * k)->NumElements(1)->Owner(true, allocator_.get());
    auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * query_num * k);
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator_->Allocate(sizeof(float) * query_num * k);
    dataset_results->Distances(dists);

    uint32_t dist_cmp = 0;

    for (int i = 0; i < query_num; ++i) {
        uint32_t temp_cmp;
        auto query_vector = query->GetSparseVectors()[i];
        this->search_one_query(query_vector, k, ids + i * k, dists + i * k, temp_cmp);
        dist_cmp += temp_cmp;
    }

    std::cout << "dist_cmp: " << dist_cmp << std::endl;
    return std::move(dataset_results);
}

void
SparseIVF::search_one_query(const SparseVector& query_vector,
                            int64_t k,
                            int64_t* res_ids,
                            float* res_dists,
                            uint32_t& dist_cmp) const {
    for (uint32_t j = 1; j < query_vector.dim_; ++j) {
        assert(query_vector.ids_[j - 1] <= query_vector.ids_[j] &&
               "IDs are not in ascending order");
    }

    MaxHeap heap(this->allocator_.get());
    auto cur_heap_top = std::numeric_limits<float>::max();

    std::unordered_set<uint32_t> visited_doc_ids;
    dist_cmp = 0;

    for (uint32_t i = 0; i < query_vector.dim_; ++i) {
        uint32_t term_id = query_vector.ids_[i];
        auto term_doc_num = this->inverted_lists_[term_id].doc_num_;
        for (uint32_t j = 0; j < term_doc_num; ++j) {
            auto doc_id = this->inverted_lists_[term_id].ids_[j];

            if (visited_doc_ids.find(doc_id) != visited_doc_ids.end()) {
                continue;
            }
            visited_doc_ids.insert(doc_id);

            SparseVector sv = this->data_[doc_id];
            float dist = SparseComputeIP(sv, query_vector);
            dist_cmp++;
            float gt = SparseComputeIPBruteForce(sv, query_vector);
            assert(std::abs(gt - dist) < 1e-1 && "IP computation has error");

            if (heap.size() < k or dist < cur_heap_top) {
                heap.emplace(dist, doc_id);
            }
            if (heap.size() > k) {
                heap.pop();
            }
            cur_heap_top = heap.top().first;
        }
    }

    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        res_dists[j] = heap.top().first;
        res_ids[j] = heap.top().second;
        heap.pop();
    }
}

}  // namespace vsag