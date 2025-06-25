
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
struct SparseIPIVFParameters{
public:
    static SparseIPIVFParameters
    FromJson(JsonType& sparse_ipivf_param_obj, IndexCommonParam index_common_param);

public:
    DocPruneStrategy doc_prune_strategy;
    VectorPruneStrategy vector_prune_strategy;
    uint32_t window_size{1};
    

protected:
    SparseIPIVFParameters() = default;
};

struct SparseIPIVFSearchParameters{
public:
    static SparseIPIVFSearchParameters
    FromJson(const std::string& json_string);

public:
    // required vars
   int num_threads{1};
   float query_cut{0.0};
   int reorder_k{0};

protected:
    SparseIPIVFSearchParameters() = default;
};

}  // namespace vsag