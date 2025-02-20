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

#include "vsag/index.h"
#include "base_filter_functor.h"
#include "common.h"
#include "safe_allocator.h"
#include "index_common_param.h"

namespace vsag{
class SparseIVF : public Index{
public:
    SparseIVF(const IndexCommonParam& index_common_param);
    ~SparseIVF(){
        this->allocator_->Deallocate(this->unique_ids);
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
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->knn_search_internal(query, k, parameters, filter));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search_internal(query, k, parameters, invalid));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {}

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {}

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {}

    virtual tl::expected<float, Error>
    CalcDistanceById(const float* vector, int64_t id) const override {
        SAFE_CALL(return alg_hnsw_->getDistanceByLabel(id, vector));
    }

    [[nodiscard]] bool
    CheckFeature(IndexFeature feature) const override;

        tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->serialize(out_stream));
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(return this->deserialize(in_stream));
    }

    int64_t
    GetNumElements() const override {
        return this->total_count_;
    }

    int64_t
    GetMemoryUsage() const override {
        SAFE_CALL(return this->cal_serialize_size());
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

    float
    calculate_distance_by_id(const float* vector, int64_t id) const;

    [[nodiscard]] BinarySet
    serialize() const;

    void
    serialize(std::ostream& out_stream) const;

    void
    serialize(StreamWriter& writer) const;

    void
    deserialize(std::istream& in_stream);

    void
    deserialize(const BinarySet& binary_set);

    void
    deserialize(const ReaderSet& reader_set);

    void
    deserialize(StreamReader& reader);

    uint64_t
    cal_serialize_size() const;

private:
    uint32_t* unique_ids;
    uint32_t total_count_{0};
    std:: shared_ptr<Allocator> allocator_{nullptr};
    std::vector<std::vector<uint32_t>> inverted_ids_lists;
    std::vector<std::vector<float>> inverted_vals_lists;
}
}