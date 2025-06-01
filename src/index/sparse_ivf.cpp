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

#include <random>

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

SparseIVF::SparseIVF(const SparseIVFParameters& param, const IndexCommonParam& index_common_param) {
    doc_prune_strategy_ = param.doc_prune_strategy;
    build_strategy_ = param.build_strategy;
    vector_prune_strategy_ = param.vector_prune_strategy;
    ivf_size_file_ = param.ivf_size_file;
    index_file_ = param.index_file;
    allocator_ = index_common_param.allocator_;
}

uint64_t
SparseIVF::cal_and_save_ivf_size() {
    std::ofstream out(ivf_size_file_, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&(this->data_dim_)), sizeof(uint32_t));
    uint64_t size = 0;
    if (this->inverted_lists_) {
        for (int i = 0; i < this->data_dim_; ++i) {
            size += this->inverted_lists_[i].doc_num_;
            out.write(reinterpret_cast<const char*>(&(this->inverted_lists_[i].doc_num_)),
                      sizeof(uint32_t));
        }
    }
    return size;
}

void 
SparseIVF::save_inverted_index() {
    std::ofstream out(index_file_, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&(this->unique_dim_)), sizeof(uint32_t));
    if (this->inverted_lists_) {
        for (uint32_t i = 0; i < this->data_dim_; ++i) {
            if(this->inverted_lists_[i].doc_num_ != 0) {
                out.write(reinterpret_cast<const char*>(&(i)),sizeof(uint32_t));
                out.write(reinterpret_cast<const char*>(&(this->inverted_lists_[i].doc_num_)),sizeof(uint32_t));
                out.write(reinterpret_cast<const char*>(this->inverted_lists_[i].ids_), this->inverted_lists_[i].doc_num_ * sizeof(uint32_t));
            }
        }
    }
}

std::vector<int64_t>
SparseIVF::build(const DatasetPtr& base) {
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

    //// build and prune word_map
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>> word_map;

    if (vector_prune_strategy_.type == VectorPruneStrategyType::VectorPrune) {
        int n_cut = vector_prune_strategy_.vectorPrune.n_cut;
        for (size_t i = 0; i < this->total_count_; ++i) {
            const SparseVector& sv = data_[i];
            std::vector<uint32_t> top_n_indices = get_top_n_indices(sv, n_cut);
            for (auto j = 0; j < std::min(n_cut, static_cast<int>(sv.dim_)); j++) {
                uint32_t word_id = sv.ids_[top_n_indices[j]];
                float val = sv.vals_[top_n_indices[j]];
                word_map[word_id].emplace_back(i, val);
            }
        }
    } else if (vector_prune_strategy_.type == VectorPruneStrategyType::NotPrune) {
        for (size_t i = 0; i < this->total_count_; ++i) {
            const SparseVector& sv = data_[i];
            for (uint32_t j = 0; j < sv.dim_; ++j) {
                uint32_t word_id = sv.ids_[j];
                word_map[word_id].emplace_back(i, sv.vals_[j]);
            }
        }
    }

    if (doc_prune_strategy_.type == DocPruneStrategyType::FixedSize) {
        fixed_pruning(word_map, doc_prune_strategy_.parameters.fixedSize.n_postings, this->data_dim_);
    } else if (doc_prune_strategy_.type == DocPruneStrategyType::GlobalPrune) {
        global_pruning(word_map, doc_prune_strategy_.parameters.globalPrune.n_postings, this->data_dim_);
        fixed_pruning(word_map,
                      doc_prune_strategy_.parameters.globalPrune.n_postings *
                          doc_prune_strategy_.parameters.globalPrune.fraction, this->data_dim_);
    }

    unique_dim_ = word_map.size();

    if (build_strategy_.type == BuildStrategyType::NotKmeans) {
        this->build_inverted_lists(word_map);
    } else if (build_strategy_.type == BuildStrategyType::Kmeans) {
        this->build_posting_lists(word_map);
    }

    if (this->ivf_size_file_ != "") {
        auto size = this->cal_and_save_ivf_size();
        std::cout << "inverted lists size: " << size << std::endl;
    }

    if (this->index_file_ != "") {
        save_inverted_index();
        std::cout << "save done!" << std::endl;
    }

    return {};
}

void
SparseIVF::build_inverted_lists(
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map) {
    //// fill in inverted_list
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
            for (uint32_t j = 0; j < doc_num; j++) {
                this->inverted_lists_[i].ids_[j] = doc_infos[j].first;
            }
        }
    }
}

void
SparseIVF::build_posting_lists(
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map) {
    //// fill in posting_lists
    this->posting_lists_ = new PostingList[this->data_dim_];

    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

#pragma omp parallel for
    for (uint32_t i = 0; i < this->data_dim_; ++i) {
        auto it = word_map.find(i);
        if (it != word_map.end()) {
            std::lock_guard<std::mutex> lock(ivf_mutex[i]);
            const auto& doc_infos = it->second;
            uint32_t doc_num = static_cast<uint32_t>(doc_infos.size());

            std::vector<uint32_t> posting_ids;
            for (uint32_t j = 0; j < doc_num; ++j) {
                posting_ids.emplace_back(doc_infos[j].first);
            }
            build_posting_list(posting_ids, i);
        }
    }
}

void
SparseIVF::build_posting_list(const std::vector<uint32_t>& posting_ids, uint32_t dim) {
    //std::cout << "dim " << dim << std::endl;
    int n_centroids = std::max(1,
                               static_cast<int>(build_strategy_.kmeans.centroid_fraction *
                                                static_cast<float>(posting_ids.size())));
    std::vector<uint32_t> reordered_posting_ids;
    std::vector<uint32_t> block_offsets;
    std::vector<std::vector<uint32_t>> clusters(n_centroids);

    if (n_centroids == 1) {
        clusters[0] = posting_ids;
    } else {
        do_kmeans_on_doc_id(posting_ids, clusters, n_centroids);
    }

    block_offsets.emplace_back(0);
    //std::cout << "cluster size" <<std::endl;
    for (auto cluster : clusters) {
        if (cluster.empty()) {
            continue;
        }
        reordered_posting_ids.insert(reordered_posting_ids.end(), cluster.begin(), cluster.end());
        block_offsets.emplace_back(reordered_posting_ids.size());
        //std::cout << cluster.size() << " ";
    }

    //std::cout << std::endl;

    assert(posting_ids.size() == reordered_posting_ids.size());
    this->posting_lists_[dim].doc_num_ = posting_ids.size();
    this->posting_lists_[dim].postings = new uint32_t[posting_ids.size()];
    memcpy(this->posting_lists_[dim].postings,
           reordered_posting_ids.data(),
           posting_ids.size() * sizeof(uint32_t));
    this->posting_lists_[dim].block_offsets = new uint32_t[block_offsets.size()];
    memcpy(this->posting_lists_[dim].block_offsets,
           block_offsets.data(),
           block_offsets.size() * sizeof(uint32_t));

    float fraction = build_strategy_.kmeans.summary_energy;

    std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>> summary;

    for (int i = 0; i < clusters.size(); ++i) {
        if (clusters[i].empty()) {
            continue;
        }
        std::vector<uint32_t> ids;
        std::vector<float> vals;
        energy_preserving_summary(ids, vals, clusters[i], fraction);
        summary.emplace_back(ids, vals);
    }

    this->posting_lists_[dim].summaries = QuantizedSummary(summary, this->data_dim_);
    this->posting_lists_[dim].num_clusters_ = block_offsets.size() - 1;
}

void
SparseIVF::do_kmeans_on_doc_id(std::vector<uint32_t> posting_ids,
                               std::vector<std::vector<uint32_t>>& clusters,
                               int n_centroids) {
    std::vector<uint32_t> centroid_ids(n_centroids);

    //// random choose n centroids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(posting_ids.begin(), posting_ids.end(), gen);
    centroid_ids.assign(posting_ids.begin(), posting_ids.begin() + n_centroids);

    for (auto doc_id : posting_ids) {
        SparseVector doc_vector = this->data_[doc_id];
        int argmin = 0;
        float min = std::numeric_limits<float>::max();

        for (int i = 0; i < n_centroids; i++) {
            auto cen_id = centroid_ids[i];
            SparseVector cen_vector = this->data_[centroid_ids[i]];
            float dist = SparseComputeIP(doc_vector, cen_vector);
            if (dist < min) {
                argmin = i;
                min = dist;
            }
        }
        clusters[argmin].emplace_back(doc_id);
    }

    std::vector<uint32_t> to_be_replaced; 

    for (int i = 0; i < n_centroids; i++) {
        if (clusters[i].size() > 0 &&
            clusters[i].size() < build_strategy_.kmeans.min_cluster_size) {
            to_be_replaced.insert(to_be_replaced.end(), clusters[i].begin(), clusters[i].end());
            clusters[i].clear();
        }
    }

    for (auto doc_id : to_be_replaced) {
        SparseVector doc_vector = this->data_[doc_id];
        int argmin = 0;
        float min = std::numeric_limits<float>::max();

        for (int i = 0; i < n_centroids; ++i) {
            if (clusters[i].empty()) {
                continue;
            }
            SparseVector cen_vector = this->data_[centroid_ids[i]];
            float dist = SparseComputeIP(doc_vector, cen_vector);
            if (dist < min) {
                argmin = i;
                min = dist;
            }
        }
        clusters[argmin].emplace_back(doc_id);
    }
}

void
SparseIVF::energy_preserving_summary(std::vector<uint32_t>& ids,
                                     std::vector<float>& vals,
                                     std::vector<uint32_t> block,
                                     float fraction) {
    std::unordered_map<uint32_t, float> hash;
    for (auto doc_id : block) {
        SparseVector sv = this->data_[doc_id];
        for (uint32_t i = 0; i < sv.dim_; ++i) {
            auto it = hash.find(sv.ids_[i]);
            if (it != hash.end()) {
                if (it->second < sv.vals_[i]) {
                    it->second = sv.vals_[i];
                }
            } else {
                hash[sv.ids_[i]] = sv.vals_[i];
            }
        }
    }

    std::vector<std::pair<uint32_t, float>> components_values(hash.begin(), hash.end());

    std::sort(components_values.begin(), components_values.end(), [](const auto& a, const auto& b) {
        return b.second < a.second;
    });

    float total_sum = std::accumulate(
        components_values.begin(), components_values.end(), 0.0f, [](float sum, const auto& pair) {
            return sum + static_cast<float>(pair.second);  // Assume T can be casted to float
        });

    float acc = 0.0f;
    for (const auto& [tid, v] : components_values) {
        acc += v;
        ids.emplace_back(tid);
        if (acc / total_sum > fraction) {
            break;
        }
    }

    std::sort(ids.begin(), ids.end());

    for (auto tid : ids) {
        vals.emplace_back(hash[tid]);
    }
}

DatasetPtr
SparseIVF::knn_search(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const std::function<bool(int64_t)>& filter) const {
    auto params = SparseIVFSearchParameters::FromJson(parameters);
    this->num_threads_ = params.num_threads;
    this->query_cut_ = params.query_cut;
    this->heap_factor_ = params.heap_factor;
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
    omp_set_num_threads(num_threads_);

    if (this->build_strategy_.type == BuildStrategyType::NotKmeans) {
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
    } else {
#pragma omp parallel for
        for (int i = 0; i < query_num; ++i) {
            std::cout << "query " << i << std::endl;
            uint32_t temp_cmp;
            auto query_vector = query->GetSparseVectors()[i];
            this->search_one_query_with_kmeans(
                query_vector, k, ids + i * k, dists + i * k, temp_cmp);
            /* #pragma omp critical
            {
                dist_cmp += temp_cmp;
            } */
        }
    }

    //std::cout << "dist_cmp: " << dist_cmp << std::endl;
    return std::move(dataset_results);
}

void
SparseIVF::search_one_query(const SparseVector& query_vector,
                            int64_t k,
                            int64_t* res_ids,
                            float* res_dists,
                            uint32_t& dist_cmp) const {
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

    MaxHeap heap(this->allocator_.get());
    auto cur_heap_top = std::numeric_limits<float>::max();

    std::unordered_set<uint32_t> visited_doc_ids;
    dist_cmp = 0;

    for (uint32_t i = 0; i < query_pair.size(); ++i) {
        uint32_t term_id = query_pair[i].first;
        auto term_doc_num = this->inverted_lists_[term_id].doc_num_;

        if (term_doc_num == 0) {
            continue;
        }

        for (uint32_t j = 0; j < term_doc_num; ++j) {
            auto doc_id = this->inverted_lists_[term_id].ids_[j];

            if (visited_doc_ids.find(doc_id) != visited_doc_ids.end()) {
                continue;
            }
            visited_doc_ids.insert(doc_id);

            SparseVector sv = this->data_[doc_id];  //shallow copy
            float dist = SparseComputeIP(sv, query_vector);
            dist_cmp++;

            if (heap.size() < k or dist < cur_heap_top) {
                heap.emplace(dist, doc_id);
            }
            if (heap.size() > k) {
                heap.pop();
            }
            cur_heap_top = heap.top().first;
        }
        /* std::cout << "term id: " << term_id << " has " << visited_doc_ids.size() << " visited doc ids : " << std::endl;
        for (const auto& id : visited_doc_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl; */
    }

    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        res_dists[j] = heap.top().first;
        res_ids[j] = heap.top().second;
        heap.pop();
    }
}

void
SparseIVF::search_one_query_with_kmeans(const SparseVector& query_vector,
                                        int64_t k,
                                        int64_t* res_ids,
                                        float* res_dists,
                                        uint32_t& dist_cmp) const {
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

    MaxHeap heap(this->allocator_.get());
    auto cur_heap_top = std::numeric_limits<float>::max();

    std::unordered_set<uint32_t> visited_doc_ids;
    dist_cmp = 0;

    for (uint32_t i = 0; i < query_pair.size(); ++i) {
        uint32_t term_id = query_pair[i].first;

        if (this->posting_lists_[term_id].doc_num_ == 0) {
            continue;
        }
        auto dots = this->posting_lists_[term_id].summaries.matmul_with_query(
            query_vector.ids_, query_vector.vals_, query_vector.dim_);

        for (auto block_id = 0; block_id < this->posting_lists_[term_id].num_clusters_;
             ++block_id) {
            if (heap.size() == k && dots[block_id] < -heap_factor_ * heap.top().first) {
                continue;
            }
            for (uint32_t j = this->posting_lists_[term_id].block_offsets[block_id];
                 j < this->posting_lists_[term_id].block_offsets[block_id + 1];
                 ++j) {
                auto doc_id = this->posting_lists_[term_id].postings[j];
                if (visited_doc_ids.find(doc_id) != visited_doc_ids.end()) {
                    continue;
                }
                visited_doc_ids.insert(doc_id);

                //std::cout << "term id: " << term_id <<" num_clusters: " << this->posting_lists_[term_id].num_clusters_ << " block id :" << block_id << " j: " << j <<std::endl;
                SparseVector sv = this->data_[doc_id];
                float dist = SparseComputeIP(sv, query_vector);
                dist_cmp++;

                if (heap.size() < k or dist < cur_heap_top) {
                    heap.emplace(dist, doc_id);
                }
                if (heap.size() > k) {
                    heap.pop();
                }
                cur_heap_top = heap.top().first;
            }
        }
    }

    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        res_dists[j] = heap.top().first;
        res_ids[j] = heap.top().second;
        heap.pop();
    }
}

void
SparseIVF::print_posting_lists() {
    //for (size_t i = 0; i < 10; ++i) {
    int i = 32;
    std::cout << "PostingList " << i << ":" << std::endl;
    std::cout << "  doc_num_: " << posting_lists_[i].doc_num_ << std::endl;

    if (posting_lists_[i].postings) {
        std::cout << "  postings: ";
        for (size_t j = 0; j < posting_lists_[i].doc_num_;
             ++j) {  // 假设 postings 有 doc_num_ 个元素
            std::cout << posting_lists_[i].postings[j] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "  postings: nullptr" << std::endl;
    }

    if (posting_lists_[i].block_offsets) {
        std::cout << "  block_offsets: ";
        for (size_t j = 0; j <= posting_lists_[i].num_clusters_;
             ++j) {  // 假设 block_offsets 有 doc_num_ 个元素
            std::cout << posting_lists_[i].block_offsets[j] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "  block_offsets: nullptr" << std::endl;
    }
    //}
}

}  // namespace vsag