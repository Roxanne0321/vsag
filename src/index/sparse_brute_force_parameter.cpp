
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

#include "sparse_brute_force_parameter.h"
#include "vsag/constants.h"

 namespace vsag {
    SparseBFParameters
    SparseBFParameters::FromJson(JsonType& sparse_bf_param_obj, IndexCommonParam index_common_param) {
        SparseBFParameters obj;
        return obj;
    }

    SparseBFSearchParameters
    SparseBFSearchParameters::FromJson(const std::string& json_string) {
        JsonType params = JsonType::parse(json_string);
        SparseBFSearchParameters obj;
        if (!params.contains(INDEX_SPARSE_BRUTE_FORCE)) {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}", INDEX_SPARSE_BRUTE_FORCE));
    }
        if(params[INDEX_SPARSE_BRUTE_FORCE].contains(SPARSE_NUM_THREADS)) {
            obj.num_threads = params[INDEX_SPARSE_BRUTE_FORCE][SPARSE_NUM_THREADS];
        }
        return obj;
    }
}