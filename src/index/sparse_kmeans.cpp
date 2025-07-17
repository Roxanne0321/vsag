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
    out_stream.write(reinterpret_cast<const char*>(&total_count_), sizeof(uint32_t));
    out_stream.write(reinterpret_cast<const char*>(&data_dim_), sizeof(uint32_t));
    out_stream.write(reinterpret_cast<const char*>(&window_num_), sizeof(uint32_t));

    for (auto dim_index = 0; dim_index < data_dim_; ++dim_index) {
        out_stream.write(reinterpret_cast<const char*>(&list_summaries_[dim_index].n_centroids_),
                         sizeof(uint32_t));
        if (list_summaries_[dim_index].n_centroids_ != 0) {
            list_summaries_[dim_index].summaries.serialize(out_stream);
        }
    }

    for (auto window_index = 0; window_index < window_num_; ++window_index) {
        auto& window = window_lists_[window_index];

        for (auto dim_index = 0; dim_index < data_dim_; ++dim_index) {
            auto n_cen = list_summaries_[dim_index].n_centroids_;
            if (n_cen != 0) {
                window[dim_index].serialize(out_stream, n_cen);
            }
        }
    }

    return {};
}

tl::expected<void, Error>
SparseKmeans::deserialize(std::istream& in_stream) {
    in_stream.read(reinterpret_cast<char*>(&total_count_), sizeof(uint32_t));
    std::cout << "total_count_: " << total_count_ << "\n";

    in_stream.read(reinterpret_cast<char*>(&data_dim_), sizeof(uint32_t));
    std::cout << "data_dim_: " << data_dim_ << "\n";

    in_stream.read(reinterpret_cast<char*>(&window_num_), sizeof(uint32_t));
    std::cout << "window_num_: " << window_num_ << "\n";

    list_summaries_.resize(data_dim_);

    for (auto dim_index = 0; dim_index < data_dim_; ++dim_index) {
        in_stream.read(reinterpret_cast<char*>(&list_summaries_[dim_index].n_centroids_),
                       sizeof(uint32_t));
        if (list_summaries_[dim_index].n_centroids_ != 0) {
            list_summaries_[dim_index].summaries.deserialize(in_stream);
        }
    }

    window_lists_.resize(window_num_);

    for (auto window_index = 0; window_index < window_num_; ++window_index) {
        auto& window = window_lists_[window_index];

        window.resize(data_dim_);
        for (auto dim_index = 0; dim_index < data_dim_; ++dim_index) {
            auto n_cen = list_summaries_[dim_index].n_centroids_;
            std::cout << "window[" << window_index << "].posting_lists_[" << dim_index << "]:\n";
            if (n_cen != 0) {
                window[dim_index].deserialize(in_stream, n_cen);
            }
        }
    }

    return {};
}

SparseKmeans::SparseKmeans(const SparseKmeansParameters& param,
                           const IndexCommonParam& index_common_param) {
    vector_prune_strategy_ = param.vector_prune_strategy;
    list_prune_strategy_ = param.list_prune_strategy;
    build_strategy_ = param.build_strategy;
    window_size_ = param.window_size;
    allocator_ = index_common_param.allocator_;
}

std::vector<int64_t>
SparseKmeans::build(const DatasetPtr& base) {
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

    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>
        word_map;  // 存储剪枝后的doc_id对应的val

    vector_prune(vector_prune_strategy_, data_, total_count_, word_map);

    list_prune(list_prune_strategy_, data_dim_, word_map);

    build_window_lists(word_map);
    return {};
}

void
SparseKmeans::build_window_lists(
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map) {
    window_num_ = total_count_ / window_size_ + ((total_count_ % window_size_ == 0) ? 0 : 1);
    window_lists_.resize(window_num_);

    // 初始化每个window下的posting_list
    for (auto window_index = 0; window_index < window_num_; ++window_index) {
        window_lists_[window_index].resize(data_dim_);
    }

    // 初始化list_summaries
    list_summaries_.resize(data_dim_);

    int min_cluster_size = build_strategy_.kmeans.min_cluster_size;
    float summary_energy = build_strategy_.kmeans.summary_energy;

#pragma omp parallel for
    for (auto i = 0; i < data_dim_; ++i) {
        if (i % 1000 == 0) {
            std::cout << "dim: " << i << std::endl;
        }
        auto it = word_map.find(i);
        if (it != word_map.end()) {
            auto dim = i;
            std::lock_guard<std::mutex> lock(ivf_mutex[i]);

            auto id_val_list = it->second;
            int n_centroids = std::max(1,
                                       static_cast<int>(build_strategy_.kmeans.centroid_fraction *
                                                        static_cast<float>(id_val_list.size())));
            std::vector<std::vector<std::pair<uint32_t, float>>> clusters(n_centroids);
            if (n_centroids == 1) {
                clusters[0] = id_val_list;
            } else {
                do_kmeans_on_doc_id(data_, id_val_list, clusters, n_centroids, min_cluster_size);
            }

            std::vector<std::vector<std::vector<std::pair<uint32_t, float>>>> window_clusters(
                window_num_);  //第一维代表窗口 第二维代表第几个cluster
            std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>> summary;

            int real_centroids = 0;  //非空cluster数目
            for (auto cluster : clusters) {
                if (!cluster.empty()) {
                    // if (dim == 100) {
                    //     std::cout << "cluster index: " << real_centroids << std::endl;
                    //     for (auto pair : cluster) {
                    //         std::cout << pair.first << " ";
                    //     }
                    //     std::cout << std::endl;
                    // }
                    real_centroids++;

                    std::vector<uint32_t> ids;
                    std::vector<float> vals;
                    std::vector<uint32_t> block_ids(cluster.size());
                    for (auto i = 0; i < cluster.size(); ++i) {
                        block_ids[i] = cluster[i].first;
                    }
                    energy_preserving_summary(data_, ids, vals, block_ids, summary_energy);
                    summary.emplace_back(ids, vals);
                }
            }

            list_summaries_[dim].n_centroids_ = real_centroids;
            list_summaries_[dim].summaries = QuantizedSummary(summary, data_dim_);

            // 每个窗口都有相同的簇数目 如果某窗口的某簇内没有分到值，则个数为0
            for (auto window_index = 0; window_index < window_num_; ++window_index) {
                window_clusters[window_index].resize(real_centroids);
            }

            int cluster_index = 0;
            for (auto cluster : clusters) {
                if (cluster.size() != 0) {
                    for (auto id_val : cluster) {
                        auto window_index = id_val.first / window_size_;
                        window_clusters[window_index][cluster_index].emplace_back(id_val);
                    }

                    cluster_index++;
                }
            }

            // if (dim == 100) {
            //     for (auto window_index = 0; window_index < window_num_; ++window_index) {
            //         std::cout << "window_index: " << window_index << std::endl;
            //         for (auto cluster_index = 0; cluster_index < real_centroids; ++cluster_index) {
            //             std::cout << "cluster_index: " << cluster_index << std::endl;
            //             for (auto pair : window_clusters[window_index][cluster_index]) {
            //                 std::cout << pair.first << " ";
            //             }
            //             std::cout << std::endl;
            //         }
            //     }
            // }

            // 对每个window下对应维度的posting list进行构建：提取该窗口下的clusters，构建PostingList
            for (auto window_index = 0; window_index < window_num_; ++window_index) {
                auto& cur_window = window_lists_[window_index];
                cur_window[dim] = PostingList(window_clusters[window_index]);
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

    uint32_t query_num = query->GetNumElements();
    auto dataset_results = Dataset::Make();
    dataset_results->Dim(query_num * k)->NumElements(1)->Owner(true, allocator_.get());
    auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * query_num * k);
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator_->Allocate(sizeof(float) * query_num * k);
    dataset_results->Distances(dists);

    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    //#pragma omp parallel for
    for (int i = 0; i < query_num; ++i) {
        auto query_vector = query->GetSparseVectors()[i];
        this->search_one_query(query_vector, k, ids + i * k, dists + i * k);
    }
    return std::move(dataset_results);
}

void
SparseKmeans::search_one_query(const SparseVector& query_vector,
                               int64_t k,
                               int64_t* res_ids,
                               float* res_dists) const {
}
}  // namespace vsag