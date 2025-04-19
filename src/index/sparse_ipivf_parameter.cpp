
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

#include "sparse_ipivf_parameter.h"

#include "vsag/constants.h"

namespace vsag {
SparseIPIVFParameters
SparseIPIVFParameters::FromJson(JsonType& sparse_ipivf_param_obj, IndexCommonParam index_common_param) {
    SparseIPIVFParameters obj;

    if (!sparse_ipivf_param_obj.contains(VECTOR_PRUNE_STRATEGY)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", VECTOR_PRUNE_STRATEGY));
    }

     if (!sparse_ipivf_param_obj[VECTOR_PRUNE_STRATEGY].contains(PRUNE_TYPE)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", PRUNE_TYPE));
    }

    std::string vector_prune_type_str = sparse_ipivf_param_obj[VECTOR_PRUNE_STRATEGY][PRUNE_TYPE];
    VectorPruneStrategy vector_prune_strat;

    if (vector_prune_type_str == "NotPrune") {
        vector_prune_strat.type = VectorPruneStrategyType::NotPrune;
    }
    else if (vector_prune_type_str == "VectorPrune") {
        vector_prune_strat.type = VectorPruneStrategyType::VectorPrune;
        vector_prune_strat.vectorPrune.n_cut = sparse_ipivf_param_obj[VECTOR_PRUNE_STRATEGY][NCUT];
    }

    obj.vector_prune_strategy = vector_prune_strat;

    if (!sparse_ipivf_param_obj.contains(DOC_PRUNE_STRATEGY)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", DOC_PRUNE_STRATEGY));
    }

     if (!sparse_ipivf_param_obj[DOC_PRUNE_STRATEGY].contains(PRUNE_TYPE)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", PRUNE_TYPE));
    }

    std::string prune_type_str = sparse_ipivf_param_obj[DOC_PRUNE_STRATEGY][PRUNE_TYPE];
    DocPruneStrategy prune_strat;

    if (prune_type_str == "NotPrune") {
        prune_strat.type = DocPruneStrategyType::NotPrune;
    } 
    else if (prune_type_str == "FixedSize") {
        prune_strat.type = DocPruneStrategyType::FixedSize;
        prune_strat.parameters.fixedSize.n_postings = sparse_ipivf_param_obj[DOC_PRUNE_STRATEGY][POSTING_LISTS];
    } 
    else if (prune_type_str == "GlobalPrune") {
        prune_strat.type = DocPruneStrategyType::GlobalPrune;
        prune_strat.parameters.globalPrune.n_postings = sparse_ipivf_param_obj[DOC_PRUNE_STRATEGY][POSTING_LISTS];
        prune_strat.parameters.globalPrune.fraction = sparse_ipivf_param_obj[DOC_PRUNE_STRATEGY][MAX_FRACTION];
    } 
    else {
        throw std::invalid_argument("Unknown strategy type");
    }
    obj.doc_prune_strategy = prune_strat;

    return obj;
}

SparseIPIVFSearchParameters
SparseIPIVFSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    SparseIPIVFSearchParameters obj;

    if (!params.contains(INDEX_SPARSE_IPIVF)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", INDEX_SPARSE_IPIVF));
    }
    
    if (params[INDEX_SPARSE_IPIVF].contains(SPARSE_NUM_THREADS)) {
        obj.num_threads = params[INDEX_SPARSE_IPIVF][SPARSE_NUM_THREADS];
    }
    if (params[INDEX_SPARSE_IPIVF].contains(QUERY_CUT)) {
        obj.query_cut = params[INDEX_SPARSE_IPIVF][QUERY_CUT];
    }
    return obj;
}
}  // namespace vsag