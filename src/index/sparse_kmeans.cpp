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

#include "sparse_kmeans.h"

#include <random>

namespace vsag {
static float
ComputeIP(const SparseVector& sv1, const SparseVector& sv2) {
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

SparseKmeans::SparseKmeans(const SparseKmeansParameters& param, const IndexCommonParam& index_common_param) {
    this->cluster_dim_size = param.cluster_dim_size;
    allocator_ = index_common_param.allocator_;
}

std::vector<int64_t>
SparseKmeans::build(const DatasetPtr& base) {
    std::unordered_map<uint32_t, float> term_val;
    this->data_dim_ = 0;
    //// copy base dataset
    const SparseVector* sparse_ptr = base->GetSparseVectors();
    this->total_count_ = base->GetNumElements();
    this->data_ = new SparseVector[this->total_count_];
    for (size_t i = 0; i < this->total_count_; ++i) {
        const SparseVector& sv = sparse_ptr[i];
        for (uint32_t j = 0; j < sv.dim_; ++j) {
            if(term_val[sv.ids_[j]] < sv.vals_[j]) {
                term_val[sv.ids_[j]] = sv.vals_[j];
            }
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
    this->unique_dim_ = term_val.size();

    std::vector<std::pair<uint32_t, float>> term_val_vector(term_val.begin(), term_val.end());

    // 进行排序
    std::sort(term_val_vector.begin(), term_val_vector.end(),
              [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                  return a.first < b.first;  // 按键升序排序
              });

    uint32_t cur = this->unique_dim_ / this->cluster_dim_size;
    auto last_size = this->unique_dim_ % this->cluster_dim_size;
    if(last_size != 0) {
        cluster_num = cur + 1;
    }
    else {
        cluster_num = cur;
    }

    this->cluster_center = new SparseVector[cluster_num];


    for(uint32_t i = 0; i < cur; ++i) {
        this->cluster_center[i].dim_ = cluster_dim_size;
        this->cluster_center[i].ids_ = new uint32_t[cluster_dim_size];
        this->cluster_center[i].vals_ = new float[cluster_dim_size];
        for (uint32_t j = 0; j < cluster_dim_size; ++j) {
            this->cluster_center[i].ids_[j] = term_val_vector[i * cluster_dim_size + j].first;
            this->cluster_center[i].vals_[j] = term_val_vector[i * cluster_dim_size + j].second;
        }
    }

    if(last_size != 0){
        auto cur_size = cur * cluster_dim_size;
        this->cluster_center[cluster_num - 1].dim_ = last_size;
        this->cluster_center[cluster_num - 1].ids_ = new uint32_t[last_size];
        this->cluster_center[cluster_num - 1].vals_ = new float[last_size];
        for (uint32_t j = 0; j < last_size; ++j) {
            this->cluster_center[cluster_num - 1].ids_[j] = term_val_vector[j].first;
            this->cluster_center[cluster_num - 1].vals_[j] = term_val_vector[j].second;
        }
    }

    std::cout << "cluster num: " << cluster_num << std::endl;

    clusters.resize(cluster_num);

    for (uint32_t doc_id = 0; doc_id < total_count_; ++doc_id) {
        const SparseVector& current_data = data_[doc_id];

        float min_distance = std::numeric_limits<float>::max();
        uint32_t closest_cluster = 0;

        for (uint32_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id) {
            const SparseVector& current_center = cluster_center[cluster_id];
            float distance = ComputeIP(current_data, current_center);

            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = cluster_id;
            }
        }

        clusters[closest_cluster].push_back(doc_id);
    }

    return {};
}

DatasetPtr
SparseKmeans::knn_search(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const std::function<bool(int64_t)>& filter) const {
    auto params = SparseKmeansSearchParameters::FromJson(parameters);
    this->num_threads_ = params.num_threads;
    this->search_num = params.search_num;
    //std::cout << "heap_factor_ is : " << heap_factor_ << std::endl;
    //std::cout << "num_threads is : " << num_threads_ << std::endl;

    uint32_t query_num = query->GetNumElements();
    auto dataset_results = Dataset::Make();
    dataset_results->Dim(query_num * k)->NumElements(1)->Owner(true, allocator_.get());
    auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * query_num * k);
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator_->Allocate(sizeof(float) * query_num * k);
    dataset_results->Distances(dists);

    uint32_t dist_cmp = 0;

    // int num_threads = omp_get_max_threads();
    omp_set_num_threads(104);

#pragma omp parallel for
        for (int i = 0; i < query_num; ++i) {
            uint32_t temp_cmp;
            auto query_vector = query->GetSparseVectors()[i];
            this->search_one_query(query_vector, k, ids + i * k, dists + i * k, temp_cmp);
            /* #pragma omp critical
            {
                dist_cmp += temp_cmp;
            } */
        }
    //std::cout << "dist_cmp: " << dist_cmp << std::endl;
    return std::move(dataset_results);
}

void
SparseKmeans::search_one_query(const SparseVector& query_vector,
                            int64_t k,
                            int64_t* res_ids,
                            float* res_dists,
                            uint32_t& dist_cmp) const {

    MaxHeap heap(this->allocator_.get());
    auto cur_heap_top = std::numeric_limits<float>::max();

    // Step 1: Calculate distance from query_vector to each cluster center
    std::vector<std::pair<float, uint32_t>> clusterDistances;
    for (uint32_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id) {
        float distance = ComputeIP(query_vector, cluster_center[cluster_id]);
        clusterDistances.emplace_back(distance, cluster_id);
    }

    // Step 2: Sort the cluster distances
    std::sort(clusterDistances.begin(), clusterDistances.end());

    // Step 3: Select search_num closest clusters
    for (uint32_t i = 0; i < search_num && i < clusterDistances.size(); ++i) {
        uint32_t cluster_id = clusterDistances[i].second;

        // Traverse through the documents in the selected cluster
        for (uint32_t doc_id : clusters[cluster_id]) {
            float dist = ComputeIP(query_vector, data_[doc_id]);

            // Use the heap to track topk closest documents
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