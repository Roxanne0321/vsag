
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
#include <omp.h>
#include <fstream>

void writeSparseMatrixToFile(const std::vector<vsag::SparseVector>& query_vectors, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);

    if (!outfile.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    int64_t row_count = query_vectors.size();
    int64_t col_count = 0;
    int64_t nnz = 0;

    for (const auto& vec : query_vectors) {
        col_count = std::max(col_count, static_cast<int64_t>(*std::max_element(vec.ids_, vec.ids_ + vec.dim_)));
        nnz += vec.dim_;
    }
    col_count++;  // Adding 1 because column indexing starts from 0

    // Writing sizes: row, col, nnz as int64
    outfile.write(reinterpret_cast<char*>(&row_count), sizeof(int64_t));
    outfile.write(reinterpret_cast<char*>(&col_count), sizeof(int64_t));
    outfile.write(reinterpret_cast<char*>(&nnz), sizeof(int64_t));

    // Prepare indptr array (int64)
    std::vector<int64_t> indptr(row_count + 1);
    int64_t current_nnz = 0;
    for (size_t i = 0; i < row_count; ++i) {
        indptr[i] = current_nnz;
        current_nnz += query_vectors[i].dim_;
    }
    indptr[row_count] = current_nnz;

    // Writing indptr
    outfile.write(reinterpret_cast<char*>(indptr.data()), indptr.size() * sizeof(int64_t));

    for (const auto& vec : query_vectors) {
        outfile.write(reinterpret_cast<char*>(vec.ids_), vec.dim_ * sizeof(uint32_t));
    }

    // Writing all values (float32) for each vector
    for (const auto& vec : query_vectors) {
        outfile.write(reinterpret_cast<char*>(vec.vals_), vec.dim_ * sizeof(float));
    }

    outfile.close();
}

void WriteSparseVectorsToBinary(const std::vector<vsag::SparseVector>& sparse_vectors, const std::string& file_name) {
    std::ofstream ofs(file_name, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening file for writing: " << file_name << std::endl;
        return;
    }

    // 写入向量数量
    uint32_t n_vecs = static_cast<uint32_t>(sparse_vectors.size());
    ofs.write(reinterpret_cast<const char*>(&n_vecs), sizeof(n_vecs));

    // 写入每个向量的数据
    for (const auto& vector : sparse_vectors) {
        // 写入向量的稀疏特征数n
        uint32_t n = vector.dim_;
        ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));

        // 写入特征索引数组（n个32位无符号整数）
        ofs.write(reinterpret_cast<const char*>(vector.ids_), n * sizeof(uint32_t));

        // 写入特征值数组（n个32位浮点数）
        ofs.write(reinterpret_cast<const char*>(vector.vals_), n * sizeof(float));
    }

    ofs.close();
}

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

std::vector<vsag::SparseVector> ReadSparseVectorsFromBinary(const std::string& file_name) {
    uint32_t n_vecs;
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.read(reinterpret_cast<char*>(&n_vecs), sizeof(n_vecs));
    std::vector<vsag::SparseVector> sparse_vectors(n_vecs);

    for (uint32_t i = 0; i < n_vecs; ++i) {
        uint32_t dim;
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        sparse_vectors[i].dim_ = dim;
        sparse_vectors[i].ids_ = new uint32_t[dim];
        sparse_vectors[i].vals_ = new float[dim];
        ifs.read(reinterpret_cast<char*>(sparse_vectors[i].ids_), dim * sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(sparse_vectors[i].vals_), dim * sizeof(float));
    }

    ifs.close();
    return sparse_vectors;
}

void PrintSparseVectors(const std::vector<vsag::SparseVector>& sparse_vectors) {
    for (const auto& vector : sparse_vectors) {
        std::cout << "Ids: ";
        for(int i = 0; i < vector.dim_; ++i) {
            std::cout << vector.ids_[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Values: ";
        for(int i = 0; i < vector.dim_; ++i) {
            std::cout << vector.vals_[i] << " ";
        }
        std::cout << std::endl;
    }
}

vsag::SparseVector*
CopyVector(const std::vector<vsag::SparseVector>& vec) {
    auto result = new vsag::SparseVector[vec.size()];
    std::memcpy(result, vec.data(), vec.size() * sizeof(vsag::SparseVector));
    return result;
}

float cal_recall(const int64_t* ids, const int64_t* gts, int64_t dim){
    float right_num = 0.0;
    for(int i = 0; i < dim; i++) {
        if(ids[i] == gts[i]) {
            right_num ++;
        }
    }
    return right_num / dim;
}

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    uint32_t size = 100;
    uint32_t max_dim = 10;
    uint32_t max_id = 20;
    float min_val = 0;
    float max_val = 1;
    int seed = 123; 

    // generate data
     std::vector<vsag::SparseVector> sparse_vectors =
        GenerateSparseVectors(size, max_dim, max_id, min_val, max_val, seed);

    WriteSparseVectorsToBinary(sparse_vectors, "data/random.bin");

    //auto sv = ReadSparseVectorsFromBinary("data/random.bin");
    //PrintSparseVectors(sv);

    auto base = vsag::Dataset::Make();
    base->SparseVectors(CopyVector(sparse_vectors))->NumElements(size)->Owner(true);


    /******************* Create SparseIVF Index *****************/
    std::string sparse_ivf_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "ip",
        "dim": 18,
        "sparse_ivf": {
            "doc_prune_strategy": {
                "prune_type": "GlobalPrune",
                "num_postings": 5,
                "max_fraction": 1.5
            },
            "build_strategy": {
                "build_type": "Kmeans",
                "centroid_fraction": 0.1,
                "min_cluster_size": 1,
                "summary_energy": 0.6
            }
        }
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
    uint32_t query_size = 100;
    uint32_t query_max_dim = 8;

    // generate data
    std::vector<vsag::SparseVector> query_vectors =
        GenerateSparseVectors(query_size, query_max_dim, max_id, min_val, max_val, seed);

    writeSparseMatrixToFile(query_vectors, "data/random_query.csr");
    auto query = vsag::Dataset::Make();
    query->SparseVectors(CopyVector(query_vectors))->NumElements(query_size)->Owner(true);

    /******************* KnnSearch For SparseIVF Index *****************/
    auto sparse_ivf_search_parameters = R"(
    {
        "sparse_ivf": {
            "query_cut": 3,
            "num_threads": 104,
            "heap_factor": 0.5
            }
        }
    )";
    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, sparse_ivf_search_parameters).value();

    /******************* KnnSearch For BruteForce Index *****************/
    std::string sparse_bf_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "ip",
        "dim": 256,
        "sparse_brute_force": {
        }
    }
    )";
    auto brute_index =
        vsag::Factory::CreateIndex("sparse_brute_force", sparse_bf_build_parameters).value();

    if (auto build_result = brute_index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index SparseIVF contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    auto sparse_bf_search_parameters = R"(
    {
        "sparse_brute_force": {
            "num_threads": 104
        }
    }
    )";

    auto brute_result = brute_index->KnnSearch(query, topk, sparse_bf_search_parameters).value();

    assert(result->GetDim() == brute_result->GetDim() && "Dim wrong");
    float recall = cal_recall(result->GetIds(), brute_result->GetIds(), result->GetDim());
    std::cout << "recall is : " << recall << std::endl; 
/*     for(int i = 0; i < result->GetDim(); ++i){
        assert(result->GetIds()[i] == brute_result->GetIds()[i] && "ids not equal");
        assert(result->GetDistances()[i] == brute_result->GetDistances()[i] && "vals not equal"); 
    } */

    return 0;
}
