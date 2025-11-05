
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

#include "sindi_parameter.h"

#include "vsag/constants.h"

namespace vsag {
SindiParameters
SindiParameters::FromJson(JsonType& sindi_param_obj, IndexCommonParam index_common_param) {
    SindiParameters obj;

    if (sindi_param_obj.contains(SINDI_LAMBDA)) {
        obj.lambda = sindi_param_obj[SINDI_LAMBDA];
    }

    if (sindi_param_obj.contains(SINDI_ALPHA)) {
        obj.alpha = sindi_param_obj[SINDI_ALPHA];
    }

    if (sindi_param_obj.contains(PRUNE_STRAGY)) {
        std::string strategy_str = sindi_param_obj[PRUNE_STRAGY];

        if (strategy_str == "FixedRatio") {
            obj.prune_stragy = PruneStrategy::FixedRatio;
        } else if (strategy_str == "MassRatio") {
            obj.prune_stragy = PruneStrategy::MassRatio;
        }
    }

    return obj;
}

SindiSearchParameters
SindiSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    SindiSearchParameters obj;

    if (!params.contains(INDEX_SINDI)) {
        throw std::invalid_argument(fmt::format("parameters must contains {}", INDEX_SINDI));
    }

    if (params[INDEX_SINDI].contains(SEARCH_NUM_THREADS)) {
        obj.num_threads = params[INDEX_SINDI][SEARCH_NUM_THREADS];
    }
    if (params[INDEX_SINDI].contains(SINDI_BETA)) {
        obj.beta = params[INDEX_SINDI][SINDI_BETA];
    }
    if (params[INDEX_SINDI].contains(SINDI_GAMMA)) {
        obj.gamma = params[INDEX_SINDI][SINDI_GAMMA];
    }

    return obj;
}
}  // namespace vsag