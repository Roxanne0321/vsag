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

#include "vsag/index.h"
#include "../utils.h"
#include "base_filter_functor.h"
#include "common.h"
#include "safe_allocator.h"
#include "typing.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "sparse_brute_force_parameter.h"
#include "algorithm/sindi/utils.h"

namespace vsag{

class SparseBF : public Index{
public:
    SparseBF(const SparseBFParameters& param, const IndexCommonParam& index_common_param);
    ~SparseBF(){
        allocator_.reset();
    }

public:
    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override {
        SAFE_CALL(return this->add(base));
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

    std::vector<int64_t>
    add(const DatasetPtr& data);

    DatasetPtr
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const std::function<bool(int64_t)>& filter) const;

    void search_one_query(const SparseVector& query_vector, int64_t k, 
                          int64_t* res_ids, float* res_dists) const;

    uint64_t
    cal_serialize_size() const {
        return 0;
    }

private:
    uint32_t total_count_{0};
    std:: shared_ptr<Allocator> allocator_{nullptr};
    SparseVector* data_;
    };
}