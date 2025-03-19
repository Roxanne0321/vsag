
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

#include "sparse_ivf_parameter.h"

#include "vsag/constants.h"

namespace vsag {
SparseIVFParameters
SparseIVFParameters::FromJson(JsonType& sparse_ivf_param_obj, IndexCommonParam index_common_param) {
    SparseIVFParameters obj;
    if (!sparse_ivf_param_obj.contains(DOC_PRUNE_STRATEGY)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", DOC_PRUNE_STRATEGY));
    }

     if (!sparse_ivf_param_obj[DOC_PRUNE_STRATEGY].contains(PRUNE_TYPE)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", PRUNE_TYPE));
    }

    std::string prune_type_str = sparse_ivf_param_obj[DOC_PRUNE_STRATEGY][PRUNE_TYPE];
    DocPruneStrategy prune_strat;

    if (prune_type_str == "NotPrune") {
        prune_strat.type = DocPruneStrategyType::NotPrune;
    } 
    else if (prune_type_str == "FixedSize") {
        prune_strat.type = DocPruneStrategyType::FixedSize;
        prune_strat.parameters.fixedSize.n_postings = sparse_ivf_param_obj[DOC_PRUNE_STRATEGY][POSTING_LISTS];
    } 
    else if (prune_type_str == "GlobalPrune") {
        prune_strat.type = DocPruneStrategyType::GlobalPrune;
        prune_strat.parameters.globalPrune.n_postings = sparse_ivf_param_obj[DOC_PRUNE_STRATEGY][POSTING_LISTS];
        prune_strat.parameters.globalPrune.fraction = sparse_ivf_param_obj[DOC_PRUNE_STRATEGY][MAX_FRACTION];
    } 
    else {
        throw std::invalid_argument("Unknown strategy type");
    }
    obj.prune_strategy = prune_strat;

    if (!sparse_ivf_param_obj.contains(BUILD_STRATEGY)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", BUILD_STRATEGY));
    }

    if (!sparse_ivf_param_obj[BUILD_STRATEGY].contains(BUILD_TYPE)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", BUILD_TYPE));
    }

    std::string build_type_str = sparse_ivf_param_obj[BUILD_STRATEGY][BUILD_TYPE];
    BuildStrategy build_strat;

    if (build_type_str == "NotKmeans") {
        build_strat.type = BuildStrategyType::NotKmeans;
    }
    else if (build_type_str == "Kmeans") {
        build_strat.type = BuildStrategyType::Kmeans;
        build_strat.kmeans.centroid_fraction = sparse_ivf_param_obj[BUILD_STRATEGY][CENTROID_FRACTION];
        build_strat.kmeans.min_cluster_size = sparse_ivf_param_obj[BUILD_STRATEGY][MIN_CLUSTER_SIZE];
        build_strat.kmeans.summary_energy = sparse_ivf_param_obj[BUILD_STRATEGY][SUMMARY_ENERGY];
    }
    
    obj.build_strategy = build_strat;
    return obj;
}

SparseIVFSearchParameters
SparseIVFSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    SparseIVFSearchParameters obj;

    if (!params.contains(INDEX_SPARSE_IVF)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", INDEX_SPARSE_IVF));
    }

    if (params[INDEX_SPARSE_IVF].contains(QUERY_CUT)) {
        obj.query_cut = params[INDEX_SPARSE_IVF][QUERY_CUT];
    }
    if (params[INDEX_SPARSE_IVF].contains(SPARSE_NUM_THREADS)) {
        obj.num_threads = params[INDEX_SPARSE_IVF][SPARSE_NUM_THREADS];
    }
    if (params[INDEX_SPARSE_IVF].contains(HEAP_FACTOR)) {
        obj.heap_factor = params[INDEX_SPARSE_IVF][HEAP_FACTOR];
    }
    return obj;
}
}  // namespace vsag