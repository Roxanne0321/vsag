
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
struct SparseBFParameters{
public:
    static SparseBFParameters
    FromJson(JsonType& sparse_bf_param_obj, IndexCommonParam index_common_param);

protected:
    SparseBFParameters() = default;
};

struct SparseBFSearchParameters{
public:
    static SparseBFSearchParameters
    FromJson(const std::string& json_string);

public:
    // required vars
   int num_threads{0};

protected:
    SparseBFSearchParameters() = default;
};

}  // namespace vsag