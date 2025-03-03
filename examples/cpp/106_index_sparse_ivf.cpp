
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

#include <vsag/vsag.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_set>

std::vector<vsag::SparseVector>
GenerateSparseVectors(
    uint32_t count, uint32_t max_dim, uint32_t max_id, float min_val, float max_val, int seed) {
    if (max_dim > static_cast<uint32_t>(max_id + 1)) {
        throw std::invalid_argument("max_dim should not exceed the total available unique ids.");
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distrib_real(min_val, max_val);

    std::vector<uint32_t> full_id_pool(max_id + 1);
    std::iota(full_id_pool.begin(), full_id_pool.end(), 0);  // 填充 [0, max_id]

    std::vector<vsag::SparseVector> sparse_vectors(count);

    for (uint32_t i = 0; i < count; ++i) {
        sparse_vectors[i].dim_ = std::uniform_int_distribution<int>(1, max_dim)(rng);

        std::shuffle(full_id_pool.begin(), full_id_pool.end(), rng);
        sparse_vectors[i].ids_ = new uint32_t[sparse_vectors[i].dim_];
        sparse_vectors[i].vals_ = new float[sparse_vectors[i].dim_];

        for (uint32_t d = 0; d < sparse_vectors[i].dim_; ++d) {
            sparse_vectors[i].ids_[d] = full_id_pool[d];
            sparse_vectors[i].vals_[d] = distrib_real(rng);
        }

        std::sort(sparse_vectors[i].ids_, sparse_vectors[i].ids_ + sparse_vectors[i].dim_);
    }

    return sparse_vectors;
}

vsag::SparseVector*
CopyVector(const std::vector<vsag::SparseVector>& vec) {
    auto result = new vsag::SparseVector[vec.size()];
    std::memcpy(result, vec.data(), vec.size() * sizeof(vsag::SparseVector));
    return result;
}

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    uint32_t size = 100;
    uint32_t max_dim = 256;
    uint32_t max_id = 300;
    float min_val = 0;
    float max_val = 100;
    int seed = 123;

    // generate data
    std::vector<vsag::SparseVector> sparse_vectors =
        GenerateSparseVectors(size, max_dim, max_id, min_val, max_val, seed);
    auto base = vsag::Dataset::Make();
    base->SparseVectors(CopyVector(sparse_vectors))->NumElements(size)->Owner(true);

    /******************* Create SparseIVF Index *****************/
    std::string sparse_ivf_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "ip",
        "dim": 256
    }
    )";
    auto index = vsag::Factory::CreateIndex("sparse_ivf", sparse_ivf_build_parameters).value();

    /******************* Build SparseIVF Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index SparseIVF contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    /******************* Prepare Query Dataset *****************/
    uint32_t query_size = 10;
    uint32_t query_max_dim = 20;

    // generate data
    std::vector<vsag::SparseVector> query_vectors =
        GenerateSparseVectors(query_size, query_max_dim, max_id, min_val, max_val, seed);
    auto query = vsag::Dataset::Make();
    query->SparseVectors(CopyVector(query_vectors))->NumElements(query_size)->Owner(true);

    /******************* KnnSearch For SparseIVF Index *****************/
    auto sparse_ivf_search_parameters = R"({})";
    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, sparse_ivf_search_parameters).value();

    /******************* KnnSearch For BruteForce Index *****************/
    auto brute_index =
        vsag::Factory::CreateIndex("sparse_brute_force", sparse_ivf_build_parameters).value();

    if (auto build_result = brute_index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index SparseIVF contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }
    auto brute_result = brute_index->KnnSearch(query, topk, sparse_ivf_search_parameters).value();

    assert(result->GetDim() == brute_result->GetDim() && "Dim wrong");
    for(int i = 0; i < result->GetDim(); ++i){
        assert(result->GetIds()[i] == brute_result->GetIds()[i] && "ids not equal");
        assert(result->GetDistances()[i] == brute_result->GetDistances()[i] && "vals not equal"); 
    }

/*     for(int i = 0; i < size; ++i) {
        delete[] sparse_vectors[i].ids_;
        delete[] sparse_vectors[i].vals_;
    }
    for(int i = 0; i < query_size; ++i) {
        delete[] query_vectors[i].ids_;
        delete[] query_vectors[i].vals_;
    } */

    return 0;
}
