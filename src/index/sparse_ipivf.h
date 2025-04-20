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

#include "../utils.h"
#include "base_filter_functor.h"
#include "common.h"
#include "safe_allocator.h"
#include "sparse_ipivf_parameter.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "typing.h"
#include "vsag/index.h"
#include "simd/simd.h"
#include "simd/fp32_simd.h"
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <mutex>
#include <fstream>

namespace vsag {
class SparseIPIVF : public Index {
public:
    SparseIPIVF(const SparseIPIVFParameters& param, const IndexCommonParam& index_common_param);
    ~SparseIPIVF() {
     if (this->inverted_lists_) {
        for (int i = 0; i < this->data_dim_; ++i) {
            if (this->inverted_lists_[i].doc_num_ != 0) {
                delete[] this->inverted_lists_[i].ids_;
                delete[] this->inverted_lists_[i].vals_;
                }
            }
        delete[] this->inverted_lists_;
     }

    for(auto &lock : ivf_mutex) {
        std::lock_guard<std::mutex> lg(lock);
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
        return {};
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
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

    std::vector<uint32_t>
    get_top_n_indices(const SparseVector& vec, uint32_t n);

    void
    fixed_pruning(std::unordered_map <uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map, int n_postings);

    void
    global_pruning(std::unordered_map <uint32_t, std::vector<std::pair<uint32_t, float>>>& word_map, int n_postings);

    DatasetPtr
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const std::function<bool(int64_t)>& filter) const;

    void
    search_one_query(const SparseVector& query_vector,
                     int64_t k,
                     int64_t* res_ids,
                     float* res_dists) const;

    void    
    multiply(std::vector<std::pair<uint32_t, float>> &query_pair, std::vector<float> &dists) const;

    void 
    scan_sort(std::vector<float> &dists, 
                int64_t k,
                int64_t* res_ids,
                float* res_dists) const;

    uint64_t
    cal_serialize_size() const {
        return 0;
    }

private:
    struct InvertedList {
        uint32_t doc_num_{0};
        uint32_t* ids_{nullptr};
        float* vals_{nullptr};
    };

    uint32_t data_dim_{0};
    uint32_t total_count_{0};
    std::shared_ptr<Allocator> allocator_{nullptr};
    InvertedList* inverted_lists_{nullptr};

//parameters
    mutable size_t query_cut_;
    mutable int num_threads_;
    DocPruneStrategy doc_prune_strategy_;
    VectorPruneStrategy vector_prune_strategy_;
    int n_postings;
    float max_fraction;
//mutex
    std::vector<std::mutex> ivf_mutex;
};
}  // namespace vsag