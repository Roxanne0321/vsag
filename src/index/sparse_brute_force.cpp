// Copyright 2024-present the vsag project
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

#include "sparse_brute_force.h"
#include <iostream>

namespace vsag {
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

SparseBF::SparseBF(const SparseBFParameter& param, const IndexCommonParam& index_common_param) {
    allocator_ = index_common_param.allocator_;
}

std::vector<int64_t>
SparseBF::build(const DatasetPtr& base) {
    return this->add(base);
}

std::vector<int64_t>
SparseBF::add(const DatasetPtr& base) {
    this->data_ = base->GetSparseVectors();
    this->total_count_ = base->GetNumElements();

    return {};
}

DatasetPtr
SparseBF::knn_search(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const std::function<bool(int64_t)>& filter) const {
    uint32_t query_num = query->GetNumElements();
    auto dataset_results = Dataset::Make();
    dataset_results->Dim(query_num * k)
        ->NumElements(1)
        ->Owner(true, allocator_.get());
    auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * query_num * k);
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator_->Allocate(sizeof(float) * query_num * k);
    dataset_results->Distances(dists);

    uint32_t dist_cmp = 0;

    for(int i = 0; i < query_num; ++i){
        uint32_t temp_cmp;
        auto query_vector = query->GetSparseVectors()[i];
        this->search_one_query(query_vector, k, ids + i * k, dists + i * k, temp_cmp);
        dist_cmp += temp_cmp;
    }

    std::cout << "dist_cmp: " << dist_cmp << std::endl;
    return std::move(dataset_results);
}

void SparseBF::search_one_query(const SparseVector& query_vector, int64_t k, 
                          int64_t* res_ids, float* res_dists, uint32_t& dist_cmp) const{
    for (uint32_t j = 1; j < query_vector.dim_; ++j) {
        assert(query_vector.ids_[j - 1] <= query_vector.ids_[j] && "IDs are not in ascending order");
    }

    MaxHeap heap(this->allocator_.get());
    float cur_heap_top = std::numeric_limits<float>::max();
    dist_cmp = 0;

    for (size_t i = 0; i < this->total_count_; ++i) {
        SparseVector base_vector = this->data_[i];

        float dist = SparseComputeIP(query_vector, base_vector);
        dist_cmp++;

        if (heap.size() < k || dist < cur_heap_top) {
            heap.emplace(dist, i);
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

}  // namespace vsag