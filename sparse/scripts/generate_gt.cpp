#include <omp.h>
#include <sys/stat.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <getopt.h>
#include <unordered_set>

#include "vsag/vsag.h"

using namespace std::chrono;

std::pair<vsag::SparseVector*, int64_t>
read_sparse_vectors_from_csr_file(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Could not open file " + filename);
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

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <basefile> <queryfile> <gtfile> <topk>\n";
        return 1;
    }

    std::string basefile = argv[1];
    std::string queryfile = argv[2];
    std::string gtfile = argv[3];
    int64_t topk = std::stoi(argv[4]);

    std::cout << "basefile: " << basefile << "\n";
    std::cout << "queryfile: " << queryfile << "\n";
    std::cout << "gtfile: " << gtfile << "\n";
    std::cout << "topk: " << topk << "\n";

    std::pair<vsag::SparseVector*, int64_t> base_results = read_sparse_vectors_from_csr_file(basefile);
    
    auto base = vsag::Dataset::Make();
    base->SparseVectors(base_results.first)->NumElements(base_results.second)->Owner(true);

    
    vsag::init();
    nlohmann::json sparse_bf_build_parameters = {
            {"dtype", "float32"},
            {"metric_type", "ip"},
            {"dim", 30000},
            {"sparse_brute_force",
             {}}};

    std::cout << "Start building Sparse Brute Force index" << std::endl;
    auto index =
        vsag::Factory::CreateIndex("sparse_brute_force", sparse_bf_build_parameters.dump()).value();

    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index Sparse Brute Force contains: "
                  << index->GetNumElements() << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }
    
    std::pair<vsag::SparseVector*, int64_t> query_results =
        read_sparse_vectors_from_csr_file(queryfile);

    auto query = vsag::Dataset::Make();
    query->SparseVectors(query_results.first)->NumElements(query_results.second)->Owner(true);

    auto sparse_bf_search_parameters = R"(
            {
                "sparse_brute_force": {
                    }
                }
            )";

    auto result = index->KnnSearch(query, topk, sparse_bf_search_parameters).value();
    std::ofstream out(gtfile, std::ios::binary);

    out.write(reinterpret_cast<const char*>(&query_results.second), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&topk), sizeof(int64_t));

    out.write(reinterpret_cast<const char*>(result->GetIds()),
                sizeof(int64_t) * topk * query_results.second);
    out.write(reinterpret_cast<const char*>(result->GetDistances()),
                sizeof(float) * topk * query_results.second);

    return 0;
}