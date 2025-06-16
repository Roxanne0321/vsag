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
tl::expected<void, Error>
SparseKmeans::serialize(std::ostream& out_stream) {
    // total_count
    out_stream.write(reinterpret_cast<const char*>(&total_count_), sizeof(total_count_));
    // data_dim
    out_stream.write(reinterpret_cast<const char*>(&data_dim_), sizeof(data_dim_));
    // max_cluster_doc_num
    out_stream.write(reinterpret_cast<const char*>(&max_cluster_doc_num_),
                     sizeof(max_cluster_doc_num_));
    // cluster_num
    out_stream.write(reinterpret_cast<const char*>(&cluster_num_), sizeof(cluster_num_));

    summaries.serialize(out_stream);

    for (auto cluster_index = 0; cluster_index < cluster_num_; cluster_index++) {
        const ClusterLists& cluster = cluster_lists_[cluster_index];
        out_stream.write(reinterpret_cast<const char*>(&cluster.doc_num_),
                         sizeof(cluster.doc_num_));
        if (cluster.doc_num_ > 0) {
            out_stream.write(reinterpret_cast<const char*>(cluster.doc_ids_.data()),
                             cluster.doc_num_ * sizeof(uint32_t));
            for (auto term_index = 0; term_index < data_dim_; ++term_index) {
                const InvertedList& list = cluster.inverted_lists_[term_index];
                out_stream.write(reinterpret_cast<const char*>(&list.doc_num_),
                                 sizeof(list.doc_num_));

                if (list.doc_num_ > 0) {
                    out_stream.write(reinterpret_cast<const char*>(list.ids_),
                                     list.doc_num_ * sizeof(uint32_t));
                    out_stream.write(reinterpret_cast<const char*>(list.vals_),
                                     list.doc_num_ * sizeof(float));
                }
            }
        }
    }
    return {};
}

tl::expected<void, Error>
SparseKmeans::deserialize(std::istream& in_stream) {
    in_stream.read(reinterpret_cast<char*>(&total_count_), sizeof(total_count_));
    in_stream.read(reinterpret_cast<char*>(&data_dim_), sizeof(data_dim_));
    in_stream.read(reinterpret_cast<char*>(&max_cluster_doc_num_), sizeof(max_cluster_doc_num_));
    in_stream.read(reinterpret_cast<char*>(&cluster_num_), sizeof(cluster_num_));

    summaries.deserialize(in_stream);

    cluster_lists_ = new ClusterLists[cluster_num_];

    for (auto cluster_index = 0; cluster_index < cluster_num_; cluster_index++) {
        ClusterLists& cluster = cluster_lists_[cluster_index];
        in_stream.read(reinterpret_cast<char*>(&cluster.doc_num_), sizeof(cluster.doc_num_));
        if (cluster.doc_num_ > 0) {
            cluster.doc_ids_.resize(cluster.doc_num_);
            in_stream.read(reinterpret_cast<char*>(cluster.doc_ids_.data()),
                           cluster.doc_num_ * sizeof(uint32_t));
            cluster.inverted_lists_ = new InvertedList[data_dim_];
            for (auto term_index = 0; term_index < data_dim_; ++term_index) {
                InvertedList& list = cluster.inverted_lists_[term_index];
                in_stream.read(reinterpret_cast<char*>(&list.doc_num_), sizeof(list.doc_num_));

                if (list.doc_num_ > 0) {
                    list.ids_ = new uint32_t[list.doc_num_];
                    list.vals_ = new float[list.doc_num_];
                    in_stream.read(reinterpret_cast<char*>(list.ids_),
                                   list.doc_num_ * sizeof(uint32_t));
                    in_stream.read(reinterpret_cast<char*>(list.vals_),
                                   list.doc_num_ * sizeof(float));
                }
            }
        }
    }

    return {};
}

SparseKmeans::SparseKmeans(const SparseKmeansParameters& param,
                           const IndexCommonParam& index_common_param) {
    this->cluster_num_ = param.cluster_num;
    this->min_cluster_size_ = param.min_cluster_size;
    this->summary_energy_ = param.summary_energy;
    this->kmeans_iter_ = param.kmeans_iter;
    allocator_ = index_common_param.allocator_;
}

std::vector<int64_t>
SparseKmeans::build(const DatasetPtr& base) {
    const SparseVector* sparse_ptr = base->GetSparseVectors();
    this->total_count_ = base->GetNumElements();
    std::vector<std::vector<uint32_t>> clusters(cluster_num_);

    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    partition_into_clusters(sparse_ptr, clusters);
    build_cluster_lists(sparse_ptr, clusters);
    return {};
}

void
SparseKmeans::partition_into_clusters(const SparseVector* sparse_ptr,
                                      std::vector<std::vector<uint32_t>>& clusters) {
    this->data_dim_ = 0;
    for (size_t i = 0; i < this->total_count_; ++i) {
        const SparseVector& sv = sparse_ptr[i];
        if (sv.ids_[sv.dim_ - 1] > this->data_dim_) {
            this->data_dim_ = sv.ids_[sv.dim_ - 1];
        }
    }

    data_dim_ += 1;

    std::vector<uint32_t> doc_ids(total_count_);
    for(auto i = 0; i < total_count_; ++i) {
        doc_ids[i] = i;
    }

    do_kmeans_on_doc_id(sparse_ptr, doc_ids, clusters, cluster_num_, min_cluster_size_, kmeans_iter_);

    // 删除数目为0的簇
    for (auto it = clusters.begin(); it != clusters.end(); ) {
        if (it->empty()) {
            it = clusters.erase(it);
        } else {
            ++it;
        }
    }

    cluster_num_ = clusters.size(); 

    // summary方式选取簇心
    std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>> summary;
    for (int i = 0; i < clusters.size(); ++i) {
        std::vector<uint32_t> ids;
        std::vector<float> vals;
        energy_preserving_summary(sparse_ptr, ids, vals, clusters[i], summary_energy_);
        summary.emplace_back(ids, vals);
    }

    summaries = QuantizedSummary(summary, data_dim_);
}

void
SparseKmeans::build_cluster_lists(const SparseVector* sparse_ptr,
                                  std::vector<std::vector<uint32_t>>& clusters) {
    this->cluster_lists_ = new ClusterLists[cluster_num_];

//#pragma omp parallel for 
    for (auto cluster_index = 0; cluster_index < cluster_num_; ++cluster_index) {
        auto cluster_doc_num = clusters[cluster_index].size();
        if (cluster_doc_num > max_cluster_doc_num_) {
            max_cluster_doc_num_ = cluster_doc_num;
        }
        ClusterLists& cluster = this->cluster_lists_[cluster_index];
        cluster.doc_num_ = cluster_doc_num;
        std::cout << "cluster.doc_num_: " << cluster.doc_num_ << std::endl;
        cluster.doc_ids_.resize(cluster_doc_num);

        // 为每个簇构建word_map
        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>> word_map;
        for (auto doc_index = 0; doc_index < cluster_doc_num; ++doc_index) {
            auto doc_id = clusters[cluster_index][doc_index];
            const SparseVector& sv = sparse_ptr[doc_id];
            for (uint32_t j = 0; j < sv.dim_; ++j) {
                uint32_t word_id = sv.ids_[j];
                float val = sv.vals_[j];
                word_map[word_id].emplace_back(doc_id, val);
            }
        }

        // 为每个word_map实现reallocate，用map实现，检查key是否存在，不存在则顺延
        std::unordered_map<uint32_t, uint32_t> doc_id_mapping;  // 原始到新的 doc_id 映射
        uint32_t next_id = 0;                                   // 当前分配的下一个新 doc_id

        for (auto& [term_id, doc_list] : word_map) {
            for (auto& [original_doc_id, value] : doc_list) {
                if (doc_id_mapping.find(original_doc_id) == doc_id_mapping.end()) {
                    // 如果原始 doc_id 尚未分配，则分配一个新的 doc_id
                    doc_id_mapping[original_doc_id] = next_id;
                    cluster.doc_ids_[next_id] = original_doc_id;
                    ++next_id;
                }
                // 替换成新的 doc_id
                original_doc_id = doc_id_mapping[original_doc_id];
            }
        }

        // 为每个cluster list构建倒排列表
        cluster.inverted_lists_ = new InvertedList[this->data_dim_];
        for (uint32_t i = 0; i < this->data_dim_; ++i) {
            auto it = word_map.find(i);
            if (it != word_map.end()) {
                auto& doc_infos = it->second;
                std::sort(doc_infos.begin(),
                          doc_infos.end(),
                          [](const std::pair<uint32_t, float>& a,
                             const std::pair<uint32_t, float>& b) { return a.first < b.first; });
                uint32_t doc_num = static_cast<uint32_t>(doc_infos.size());
                cluster.inverted_lists_[i].doc_num_ = doc_num;
                cluster.inverted_lists_[i].ids_ = new uint32_t[doc_num];
                cluster.inverted_lists_[i].vals_ = new float[doc_num];
                for (uint32_t j = 0; j < doc_num; j++) {
                    cluster.inverted_lists_[i].ids_[j] = doc_infos[j].first;
                    cluster.inverted_lists_[i].vals_[j] = doc_infos[j].second;
                }
            }
        }
    }
}

DatasetPtr
SparseKmeans::knn_search(const DatasetPtr& query,
                         int64_t k,
                         const std::string& parameters,
                         const std::function<bool(int64_t)>& filter) const {
    auto params = SparseKmeansSearchParameters::FromJson(parameters);
    this->num_threads_ = params.num_threads;
    this->search_num_ = params.search_num;

    uint32_t query_num = query->GetNumElements();
    auto dataset_results = Dataset::Make();
    dataset_results->Dim(query_num * k)->NumElements(1)->Owner(true, allocator_.get());
    auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * query_num * k);
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator_->Allocate(sizeof(float) * query_num * k);
    dataset_results->Distances(dists);

    uint32_t dist_cmp = 0;

    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    long long search_data_num = 0;
    long long accumulation_time = 0;
    long long heap_time = 0;

//#pragma omp parallel for
    for (int i = 0; i < query_num; ++i) {
        auto query_vector = query->GetSparseVectors()[i];
        this->search_one_query(query_vector, k, ids + i * k, dists + i * k, search_data_num, accumulation_time, heap_time);
    }

    std::cout << "search_data_num: " << search_data_num / query_num << std::endl;
    std::cout << "accumulation_time: " << accumulation_time << std::endl;
    std::cout << "heap_time: " << heap_time << std::endl;
    return std::move(dataset_results);
}

void
SparseKmeans::search_one_query(const SparseVector& query_vector,
                               int64_t k,
                               int64_t* res_ids,
                               float* res_dists,
                               long long &search_data_num,
                               long long &accumulation_time,
                               long long &heap_time) const {
    MaxHeap heap(this->allocator_.get());
    auto cur_heap_top = std::numeric_limits<float>::max();

    std::vector<float> results = summaries.matmul_with_query(query_vector.ids_, query_vector.vals_, query_vector.dim_);

    std::vector<size_t> indices(results.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    std::nth_element(indices.begin(),
                     indices.begin() + search_num_,
                     indices.end(),
                     [&results](size_t a, size_t b) { return results[a] > results[b]; });

    std::vector<float> dists(max_cluster_doc_num_, 0.0);

    for (auto i = 0; i < search_num_; ++i) {
        if(cluster_lists_[indices[i]].doc_num_ != 0) {
            search_data_num += cluster_lists_[indices[i]].doc_num_;
            search_one_cluster(query_vector, indices[i], dists, k, heap, cur_heap_top, accumulation_time, heap_time);
        }
    }

    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        res_dists[j] = heap.top().first;
        res_ids[j] = heap.top().second;
        heap.pop();
    }
}

void
SparseKmeans::search_one_cluster(const SparseVector& query_vector,
                                 uint32_t cluster_id,
                                 std::vector<float>& dists,
                                 int64_t k,
                                 MaxHeap& heap,
                                 float cur_heap_top,
                                 long long &accumulation_time,
                                 long long &heap_time) const {
    ClusterLists& cluster = cluster_lists_[cluster_id];
    auto start_time_1 = std::chrono::high_resolution_clock::now();
    for (auto term_index = 0; term_index < query_vector.dim_; ++term_index) {
        auto term_id = query_vector.ids_[term_index];
        float query_val = -query_vector.vals_[term_index];
        const InvertedList& list = cluster.inverted_lists_[term_id];
        if (list.doc_num_ == 0) [[unlikely]] {
            continue;
        }
        for (auto doc_id_index = 0; doc_id_index < list.doc_num_; ++doc_id_index) {
            auto doc_id = list.ids_[doc_id_index];
            dists[doc_id] += list.vals_[doc_id_index] * query_val;
        }
    }
    auto end_time_1 = std::chrono::high_resolution_clock::now();
    accumulation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_1 - start_time_1).count();

    // 记着dists归零
    // 入堆时按照堆内记录的doc数目入堆
    auto start_time_2 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < cluster.doc_num_; ++i) {
        if (dists[i] >= cur_heap_top) [[likely]] {
            dists[i] = 0;
            continue;
        } else {
            heap.emplace(dists[i], cluster.doc_ids_[i]);
        }
        if (heap.size() > k) {
            heap.pop();
        }
        cur_heap_top = heap.top().first;
        dists[i] = 0;
    }
    auto end_time_2 = std::chrono::high_resolution_clock::now();
    heap_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_2 - start_time_2).count();

}
}  // namespace vsag