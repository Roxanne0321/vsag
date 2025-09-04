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

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <basefile> <lambda> <alpha> <index_path>\n";
        return 1;
    }

    std::string basefile = argv[1];
    int lambda = std::stoi(argv[2]);
    float alpha = std::stof(argv[3]);
    std::string index_path = argv[4];

    std::cout << "basefile: " << basefile << "\n";
    std::cout << "lambda: " << lambda << "\n";
    std::cout << "alpha: " << alpha << "\n";
    std::cout << "index_path: " << index_path << "\n";

    std::pair<vsag::SparseVector*, int64_t> base_results = read_sparse_vectors_from_csr_file(basefile);
    
    auto base = vsag::Dataset::Make();
    base->SparseVectors(base_results.first)->NumElements(base_results.second)->Owner(true);

    
    vsag::init();
    nlohmann::json sindi_build_parameters = {
            {"dtype", "float32"},
            {"metric_type", "ip"},
            {"dim", 30000},
            {"sindi",
             {{"lambda", lambda},
              {"alpha", alpha}}}};

    std::cout << "Start building sindi index" << std::endl;
    auto index =
        vsag::Factory::CreateIndex("sindi", sindi_build_parameters.dump()).value();

    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index Sindi contains: "
                  << index->GetNumElements() << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    std::ofstream index_file(index_path, std::ios::binary);

    if (!index_file) {
        std::cerr << "Error opening file for serialization." << std::endl;
        return 1;
    }
    index->Serialize(index_file);

    return 0;
}