
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
#include "parameter.h"
#include "typing.h"
#include "index_common_param.h"

namespace vsag {
struct SparseKmeansParameters{
public:
    static SparseKmeansParameters
    FromJson(JsonType& sparse_kmeans_param_obj, IndexCommonParam index_common_param);

public:
    uint32_t window_size;
    ListPruneStrategy list_prune_strategy;
    VectorPruneStrategy vector_prune_strategy;
    BuildStrategy build_strategy;

protected:
    SparseKmeansParameters() = default;
};

struct SparseKmeansSearchParameters{
public:
    static SparseKmeansSearchParameters
    FromJson(const std::string& json_string);

public:
    float query_cut;
    int num_threads;
    int reorder_k;
    float heap_factor;

protected:
    SparseKmeansSearchParameters() = default;
};
}  // namespace vsag