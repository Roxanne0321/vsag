
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
#include <sys/stat.h>

float cal_recall(const int64_t* ids, const int32_t* gts, int64_t dim){
    float right_num = 0.0;
    for(int i = 0; i < dim; i++) {
        if(ids[i] == gts[i]) {
            right_num ++;
        }
    }
    return right_num / dim;
}

std::vector<int32_t> knn_result_read(const std::string& fname) {
    std::ifstream file(fname, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot open file: " << fname << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 读取前两个 uint32 值
    uint32_t n, d;
    file.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));

    // 确认文件大小
    struct stat file_stat;
    if (stat(fname.c_str(), &file_stat) != 0) {
        std::cerr << "Cannot get file size: " << fname << std::endl;
        std::exit(EXIT_FAILURE);
    }
    size_t expected_size = 8 + n * d * (4 + 4);  // 8 bytes header, n*d*(4+4) for data
    assert(file_stat.st_size == expected_size);

    // 读取 ID 矩阵并转化为一维向量
    std::vector<int32_t> I(n * d);
    file.read(reinterpret_cast<char*>(I.data()), n * d * sizeof(int32_t));    

    // 关闭文件
    file.close();

    return I;
}

std::pair<vsag::SparseVector*, int64_t>
read_sparse_vectors_from_csr_file(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            throw std::runtime_error("Could not open file");
        }

        int64_t sizes[3];
        infile.read(reinterpret_cast<char*>(sizes), 3 * sizeof(int64_t));
        int64_t num_rows = sizes[0];
        int64_t num_cols = sizes[1];
        int64_t nnz = sizes[2];

        std::vector<int64_t> indptr(num_rows + 1);
        infile.read(reinterpret_cast<char*>(indptr.data()), (num_rows + 1) * sizeof(int64_t));

        std::vector<int32_t> indices(nnz);
        infile.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int32_t));

        std::vector<float> data(nnz);
        infile.read(reinterpret_cast<char*>(data.data()), nnz * sizeof(float));

        infile.close();
    
    vsag::SparseVector* sparse_vectors = new vsag::SparseVector[num_rows];

        for (int64_t i = 0; i < num_rows; ++i) {
            int64_t row_start = indptr[i];
            int64_t row_end = indptr[i + 1];
            int64_t row_size = row_end - row_start;

            sparse_vectors[i].dim_ = static_cast<uint32_t>(row_size);
            sparse_vectors[i].ids_ = new uint32_t[row_size];
            sparse_vectors[i].vals_ = new float[row_size];

            std::memcpy(
                sparse_vectors[i].ids_, indices.data() + row_start, row_size * sizeof(uint32_t));
            std::memcpy(sparse_vectors[i].vals_, data.data() + row_start, row_size * sizeof(float));
        }
        return std::make_pair(sparse_vectors, num_rows);
}

int
main(int argc, char** argv) {
    vsag::init();

    std::string basefile = "data/safe/bge_safe_doc.csr";

    std::pair<vsag::SparseVector*, int64_t> base_results;

    base_results = read_sparse_vectors_from_csr_file(basefile);

    auto base = vsag::Dataset::Make();
    base->SparseVectors(base_results.first)->NumElements(base_results.second)->Owner(true);


    /******************* Create SparseIVF Index *****************/
    std::string sparse_ivf_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "ip",
        "dim": 30000,
        "sparse_ivf": {
            "doc_prune_strategy": {
                "prune_type": "NotPrune"
            },
            "build_strategy": {
               "build_type": "NotKmeans"
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

    std::string queryfile = "data/safe/bge_safe_query.csr";

    std::pair<vsag::SparseVector*, int64_t> query_results;

    query_results = read_sparse_vectors_from_csr_file(queryfile);

    auto query = vsag::Dataset::Make();
    query->SparseVectors(query_results.first)->NumElements(query_results.second)->Owner(true);

     auto sparse_ivf_search_parameters = R"(
    {
        "sparse_ivf": {
            "query_cut": 0,
            "num_threads": 104,
            "heap_factor": 0
            }
        }
    )";
    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, sparse_ivf_search_parameters).value();

    std::string gtfile = "data/safe/bge_safe_recall.dev.gt";

    auto gt_results = knn_result_read(gtfile);

    float recall = cal_recall(result->GetIds(), gt_results.data(), result->GetDim());
    std::cout << "recall is : " << recall << std::endl;

    return 0;
}
