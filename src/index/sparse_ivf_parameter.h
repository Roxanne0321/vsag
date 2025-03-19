
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
enum class DocPruneStrategyType {
    NotPrune,
    FixedSize,
    GlobalPrune
};

struct FixedSize {
    int n_postings;
};

struct GlobalPrune {
    int n_postings;
    float fraction;
};

struct DocPruneStrategy {
    DocPruneStrategyType type;
    union {
        FixedSize fixedSize;
        GlobalPrune globalPrune;
    } parameters;
};

enum class BuildStrategyType {
    NotKmeans,
    Kmeans
};

struct Kmeans {
    int min_cluster_size;
    float centroid_fraction;
    float summary_energy;
};

struct BuildStrategy {
    BuildStrategyType type;
    Kmeans kmeans;
};

struct SparseIVFParameters{
public:
    static SparseIVFParameters
    FromJson(JsonType& sparse_ivf_param_obj, IndexCommonParam index_common_param);

public:
    DocPruneStrategy prune_strategy;
    BuildStrategy build_strategy;

protected:
    SparseIVFParameters() = default;
};

struct SparseIVFSearchParameters{
public:
    static SparseIVFSearchParameters
    FromJson(const std::string& json_string);

public:
    // required vars
   int num_threads{1};
   float heap_factor{0.0};
   size_t query_cut{0};

protected:
    SparseIVFSearchParameters() = default;
};

}  // namespace vsag