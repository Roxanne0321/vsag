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

#pragma once

#include <omp.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>

#include "../utils.h"
#include "common.h"
#include "safe_allocator.h"
#include "typing.h"
#include "vsag/index.h"
#include "sparse_kmeans_parameters.h"
#include "algorithm/seismic/utils.h"
#include "algorithm/seismic/summary.h"

namespace vsag {
class SparseKmeans : public Index {
public:
    SparseKmeans(const SparseKmeansParameters& param, const IndexCommonParam& index_common_param);
    ~SparseKmeans() {
        if (this->cluster_lists_) {
            for (int i = 0; i < cluster_num_; ++i) {
                if (cluster_lists_[i].inverted_lists_) {
                    for (int j = 0; j < this->data_dim_; ++j) {
                        if (this->cluster_lists_[i].inverted_lists_[j].doc_num_ != 0) {
                            delete[] this->cluster_lists_[i].inverted_lists_[j].ids_;
                            delete[] this->cluster_lists_[i].inverted_lists_[j].vals_;
                        }
                    }
                    delete[] this->cluster_lists_[i].inverted_lists_;
                }
            }
            delete[] this->cluster_lists_;
        }

        allocator_.reset();
    }

public:
    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        std::function<bool(int64_t)> func = [&invalid](int64_t id) -> bool {
            int64_t bit_index = id & ROW_ID_MASK;
            return invalid->Test(bit_index);
        };
        if (invalid == nullptr) {
            func = nullptr;
        }
        SAFE_CALL(return this->knn_search(query, k, parameters, func));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, filter));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        return {};
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        return {};
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        return {};
    }

    tl::expected<BinarySet, Error>
    Serialize() const override {
        BinarySet bs;
        return bs;
    }

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->serialize(out_stream));
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(return this->deserialize(in_stream));
        return {};
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        return {};
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        return {};
    }

    int64_t
    GetNumElements() const override {
        return this->total_count_;
    }

    int64_t
    GetMemoryUsage() const override {
        return this->cal_serialize_size();
    }

private:
    std::vector<int64_t>
    build(const DatasetPtr& data);

    void
    partition_into_clusters(const SparseVector* sparse_ptr,
                            std::vector<std::vector<uint32_t>>& clusters);

    void
    build_cluster_lists(const SparseVector* sparse_ptr,
                        std::vector<std::vector<uint32_t>>& clusters);

    DatasetPtr
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const std::function<bool(int64_t)>& filter) const;

    void
    search_one_query(const SparseVector& query_vector,
                     int64_t k,
                     int64_t* res_ids,
                     float* res_dists,
                     long long &search_data_num,
                     long long &accumulation_time,
                     long long &heap_time) const;

    void
    search_one_cluster(const SparseVector& query_vector,
                       uint32_t cluster_id,
                       std::vector<float> &dists,
                       int64_t k,
                       MaxHeap &heap,
                       float cur_heap_top,
                       long long &accumulation_time,
                       long long &heap_time) const;

    uint64_t
    cal_serialize_size() const {
        return 0;
    }

    tl::expected<void, Error>
    serialize(std::ostream& out_stream);

    tl::expected<void, Error>
    deserialize(std::istream& in_stream);

private:
    struct InvertedList {
        uint32_t doc_num_{0};
        uint32_t* ids_{nullptr};
        float* vals_{nullptr};
    };

    struct ClusterLists {
        uint32_t doc_num_{0};
        std::vector<uint32_t> doc_ids_;
        InvertedList* inverted_lists_{nullptr};
    };

    uint32_t data_dim_{0};
    uint32_t total_count_{0};
    uint32_t max_cluster_doc_num_{0};
    std::shared_ptr<Allocator> allocator_{nullptr};

    uint32_t cluster_num_;
    uint32_t min_cluster_size_;
    float summary_energy_;
    uint32_t kmeans_iter_;
    ClusterLists* cluster_lists_{nullptr};
    QuantizedSummary summaries;

    //parameters
    mutable int num_threads_;
    mutable uint32_t search_num_;
};
}  // namespace vsag