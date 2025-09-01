
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

#include "sparse_kmeans_parameters.h"

#include "vsag/constants.h"

namespace vsag {
SparseKmeansParameters
SparseKmeansParameters::FromJson(JsonType& sparse_kmeans_param_obj, IndexCommonParam index_common_param) {
    SparseKmeansParameters obj;

    obj.cluster_num = sparse_kmeans_param_obj[INDEX_CLUSTER_NUM];
    obj.min_cluster_size = sparse_kmeans_param_obj[MIN_CLUSTER_SIZE];
    obj.summary_energy = sparse_kmeans_param_obj[SUMMARY_ENERGY];
    obj.kmeans_iter = sparse_kmeans_param_obj[KMEANS_ITER];
    return obj;
}

SparseKmeansSearchParameters
SparseKmeansSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    SparseKmeansSearchParameters obj;

   obj.search_num = params[INDEX_SPARSE_KMEANS][INDEX_SEARCH_NUM];
    return obj;
}
}  // namespace vsag